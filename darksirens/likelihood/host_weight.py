"""Per-host log-weights for the host-sum PE estimator.

Produces the ``ldw`` array the existing per-event reduction already consumes
(``log_evidence_and_mc_variance``: ``-log n + logsumexp(ldw)``), so the block
plan, the scan, the -inf masking contract and the variance guard are all
reused unchanged.  Only the CONTENT of ldw differs:

    sampled path :  log p_pop + log p_cat(z_s) - log J - log p_pe(s)
    host path    :  log wbar_g + log Z_i,g - log J

with the per-host term a single sum over the event's PE samples::

    Z_i,g = (1/N) sum_s  K(dL(z_g,H0) - dL_s)
                         * p_pop(m1det_s/(1+z_g), q_s, z_g, chi_s) / p_pe(s)

Only dL carries a kernel; the mass/spin coordinates are used at their sample
values.  The product stays inside the sum -- factorising it into
``<K>_s * <p_pop>_s`` costs up to 5.2 nats (see ``host_log_weights``).

Note what is absent.  ``log_prior_z`` does NOT appear: summing over catalogued
hosts is what evaluating p_cat was approximating, so evaluating it here too
would count the catalog twice.  p_pe appears only as the 1/p_pe reweighting
inside Z_i,g -- the PE enters as a LIKELIHOOD, not as a proposal density the
redshift integral divides out.

The selection term is deliberately untouched by all of this: it is tabulated
(``--beta_table``) and its target drops p_z(z|pix) by design, since mu is a
sky-averaged normalisation.  Do not "fix" that asymmetry.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from ..inference.utils import log_jacobian_m1src_q_z_to_m1det_q_dL

_HALF_LOG_2PI: float = 0.9189385332046727


def pe_sample_weights(dL_s, log_p_pe_s, bandwidth=0.15):
    """Per-sample log weights and the distance-kernel inverse bandwidth.

    The stored samples are drawn from the posterior ~ L_i(dL) p_pe, so each is
    weighted by 1/p_pe to recover the likelihood.  The kernel itself is applied
    inside ``host_log_weights``, per sample, so the (dL, mass) pairing survives
    -- see that docstring.  Bandwidth is a fraction of the weighted sample
    spread; results are insensitive over 0.08-0.30.
    """
    log_w = -log_p_pe_s
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    w = jnp.exp(log_w)
    mu = jnp.sum(w * dL_s)
    var = jnp.sum(w * (dL_s - mu) ** 2)
    inv_h = 1.0 / jnp.maximum(bandwidth * jnp.sqrt(jnp.maximum(var, 1e-12)),
                              1e-300)
    return log_w, inv_h


def host_log_weights(
    z_host,
    log_w_host,
    dL_s,
    m1det_s,
    q_s,
    chieff_s,
    log_p_pe_s,
    cosmo,
    pop_params,
    log_p_pop_fn,
    zgrid=None,
    dL_grid=None,
    ddL_grid=None,
    pop_z_horizon=jnp.inf,
    spin=None,
    kde_bandwidth=0.15,
    host_chunk=128,
):
    """``ldw`` for ONE event's hosts: (n_draw,) log-weights.

    MIXED PE REPRESENTATION.  Each term of the inner sum is ONE PE sample,
    carrying its own masses and spin, weighted by a distance kernel centred on
    its own dL::

        Z_i,g = (1/N) sum_s  K(dL(z_g,H0) - dL_s)
                             * p_pop(m1det_s/(1+z_g), q_s, z_g, chi_s)
                             / p_pe(s)

    ONLY dL gets a kernel.  The integrand is evaluated at dL(z_g, H0) -- a
    point set by the host, not by any stored sample -- so the distance
    likelihood must be interpolated BETWEEN samples.  m1det/q/chi_eff need no
    kernel: p_pop is analytic and is evaluated at each sample's own value.

    THE PRODUCT SITS INSIDE THE SUM.  An earlier version averaged the kernel
    and the population factor separately and multiplied them,
    ``<K>_s * <p_pop>_s``, which is exact only if dL is independent of the mass
    sector.  It is not: likelihood-weighted r(dL, m1det) reaches 0.488 on this
    dataset (9/44 events above 0.3, q anti-correlated similarly), and
    factorising cost up to 5.2 nats of shape error in log Z_i(H0) -- against a
    0.02-nat agreement gate (job 1210295).  Do not re-factorise this for speed.

    chi_eff carries per-sample and has no (1+z) coupling, so its p_pop factor
    is constant across the hosts of one event: it cannot distort the H0 shape,
    however it correlates with dL, provided it stays inside the sum.

    Cost is (n_draw x nsamp) per event -- unchanged by the fix, since the same
    grid was already being built.
    """
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    # dL(z_g) by interpolating the SAME (zgrid, dL_grid) pair the sampled path
    # inverts, so both estimators sit on one distance-redshift relation.
    dL_at = jnp.interp(z_host, zgrid, dL_grid)

    log_w_s, inv_h = pe_sample_weights(dL_s, log_p_pe_s, bandwidth=kde_bandwidth)
    log_knorm = jnp.log(inv_h) - _HALF_LOG_2PI

    # Chunked over hosts.  Dense (n_draw, nsamp) is only 0.13 GB at 4000x4000,
    # but XLA fuses it into multi-copy intermediates across the event vmap:
    # MEASURED 200x the base array (190.75 GiB asked for a 1.02 GB block=8
    # grid, job 1181188).  One (host_chunk, nsamp) tile stays live instead.
    n_h = z_host.shape[0]
    chunk = int(min(max(host_chunk, 1), n_h))
    n_full = n_h // chunk
    rem = n_h - n_full * chunk

    def _tile(z_tile, d_tile):
        zz = z_tile[:, None]
        m1src = m1det_s[None, :] / (1.0 + zz)
        z_b = jnp.broadcast_to(zz, m1src.shape)
        q_b = jnp.broadcast_to(q_s[None, :], m1src.shape)
        ce_b = jnp.broadcast_to(chieff_s[None, :], m1src.shape)
        if spin is None:
            lp = log_p_pop_fn(m1src, q_b, z_b, ce_b, pop_params)
        else:
            lp = log_p_pop_fn(m1src, q_b, z_b, ce_b, pop_params, spin=spin)
        u = (d_tile[:, None] - dL_s[None, :]) * inv_h
        log_kern = -0.5 * u * u + log_knorm
        # Kernel and population factor multiplied per sample, THEN summed.
        return jax.scipy.special.logsumexp(
            log_w_s[None, :] + log_kern + lp, axis=1)

    parts = []
    if n_full:
        zt = z_host[:n_full * chunk].reshape(n_full, chunk)
        dt = dL_at[:n_full * chunk].reshape(n_full, chunk)
        parts.append(
            lax.map(lambda a: _tile(a[0], a[1]), (zt, dt)).reshape(-1))
    if rem:
        parts.append(_tile(z_host[n_full * chunk:], dL_at[n_full * chunk:]))
    log_Z_g = jnp.concatenate(parts) if len(parts) > 1 else parts[0]

    log_J = log_jacobian_m1src_q_z_to_m1det_q_dL(
        z_host, dL_at, H0, Om0, w0, wa, ddL_grid=ddL_grid
    )

    ldw = log_w_host + log_Z_g - log_J
    supported = jnp.isfinite(log_w_host) & (z_host <= pop_z_horizon)
    return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)