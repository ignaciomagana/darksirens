"""P17 -- the Gaussian-marginalization limit (PLAN sec 1.6 Limit III, sec 6.3, sec 6.5).

This is the ONLY test in the ladder that validates the member-marginalization
estimator against a CLOSED FORM rather than against a larger ``M_draw``.  Every
other marginalization pin (``test_marginalize_equals_logmeanexp_of_members``,
``tests/test_latent_seam_e2e.py::E1``) is a self-consistency statement: it
checks that the estimator equals itself computed a different way.  P17 checks
that the number the estimator produces is the number the mathematics says it
must produce, in a corner where the mathematics is exactly solvable.

That corner is the small-``b_GW`` limit.  Marginalizing a Gaussian latent field
analytically REPLACES THE FIELD BY THE KERNEL: expand ``ll(xi)`` to first order
in ``b_GW (Phi xi)``, marginalize ``xi ~ N(0, I)``, and

    log int dxi N(xi; 0, I) exp[a . xi]  =  (1/2) ||a||^2 ,

so the log evidence gain is a pure contraction of ``K = Phi Phi^T`` over the
events' positions.  This is the same algebra that produces Cheng & Gair's
cross-correlation statistic as a limit of this hierarchy (PLAN sec 1.6), and the
same first-order expansion that produces sec 6.5 eq. (6)'s closed-form member
spread.  If the shipped ``logsumexp_m(ll_m) - log M`` reproduces it, the
estimator is validated against truth at the one amplitude where truth is
available.

----------------------------------------------------------------------------
WHAT THE CLOSED FORM ACTUALLY IS -- and where PLAN sec 1.6 is incomplete
----------------------------------------------------------------------------

PLAN sec 1.6 and the P17 row of sec 6.3 state the arm-(a) target as

    LSE_m ll_m - log M  ->  (b_GW^2 / 2) sum_{ij} K(x_i, x_j)                (P)

with ``a_GW = b_GW sum_i Phi_i``.  **That form is not what the shipped seam
converges to, and the difference is not small.**  It drops the budget
normalizer.  ``latent_q.rho_from_moments`` subtracts, at every ``z``,

    rho_m(z) = log[ (A_m - c B_m) / (P_F - c F_F) ]
             = log[ (1/P_F) sum_{p in F} e^{b_GW u_m(p, z)} ]      (at c = 0)

with ``u_m(p, z) = Phi_{(p,z)} . xi_m`` -- the footprint MONOPOLE of the field
at that ``z``.  It is not optional: it is what makes PLAN eq. (4)'s consumed
budget identity hold, and it is present in every latent evaluation.  Carrying
it through the same expansion gives, per event ``i`` at position
``x_i = (p_i, z_i)``,

    logQ_m(x_i) = b_GW [ u_m(x_i) - <u_m(., z_i)>_F ]
                  - (b_GW^2 / 2) Var_{p in F}( u_m(p, z_i) )  + O(b_GW^3)

so BOTH orders are modified.  The first-order term contracts the MONOPOLE-
PROJECTED basis ``psi_i = Phi(x_i) - <Phi(., z_i)>_F``; the second-order term is
a deterministic (xi-independent in expectation) self-energy.  Marginalizing:

    LSE_m ll_m - log M - ll(xi = 0)  ->  b_GW^2 * c_inf  + O(b_GW^4)         (T)

    c_inf = (1/2) sum_{ij} Ktilde(x_i, x_j)  -  (1/2) sum_i V(z_i)           (T')

    Ktilde(x_i, x_j) = <psi_i, psi_j>
        = K(x_i, x_j)
          - (1/P_F) sum_{p in F} K(x_i, (p, z_j))
          - (1/P_F) sum_{q in F} K((q, z_i), x_j)
          + (1/P_F^2) sum_{p,q in F} K((q, z_i), (p, z_j))

    V(z) = (1/P_F) sum_{p in F} K((p,z),(p,z)) - (1/P_F^2) sum_{p,q} K((p,z),(q,z))

Every object in (T') is still a pure contraction of the kernel the latent field
was built from -- "marginalizing the Gaussian field replaces the field by the
kernel" survives verbatim.  What does not survive is the naive pairing.  The
two corrections have a clean physical reading: the monopole projection removes
the field mode the budget identity conserves, and ``-V/2`` cancels the events'
SELF terms ``K(x_i, x_i)`` (a common multiplicative normalizer cannot inform an
event about its own pixel), so at large ``|F|`` (T') collapses to the strictly
OFF-DIAGONAL kernel sum ``(1/2) sum_{i != j} K(x_i, x_j)``.

**Measured, in the configuration below** (8 events, ``|F| = 14``, ``M = 512``):

    sum_{ij} Ktilde(x_i, x_j)  =  9.333742
    sum_i V(z_i)               =  3.249350
    c_inf   (T', the seam's closed form)   =  +3.042196
    c_naive (P, PLAN sec 1.6 as written)   = +26.931816     <- 8.9x too large

    measured  D(b) := [LSE_m ll_m - log M - ll(0)] / b^2  at b = 0.03536:
                                           =  +3.461972

``c_naive`` is **64 MC standard errors** away from the measured limit and is
refuted by ``test_p17_refutes_the_unprojected_kernel_form``.  This is a finding
about the PLAN's stated form, not about the code: the seam is right and sec 1.6's
one-line derivation is missing the normalizer.  It has a direct consequence for
sec 6.5 eq. (6), which is written with the same unprojected ``Phi_i``: the
member spread ``sigma = ||L_H^{-1} a||_2`` is over-predicted by whatever the
monopole projection removes, on TOP of the Euclidean-vs-``H^{-1}`` over-
prediction v4 already corrected.  Both errors push the same way -- towards an
exponentially larger ``M_draw`` requirement than the estimator actually needs.

----------------------------------------------------------------------------
WHY THE CONVERGENCE RATE IS b^4, AND WHY THAT IS THE REAL TEST
----------------------------------------------------------------------------

A single-point agreement proves little: any estimator can be accidentally right
at one amplitude.  The RATE is the discriminating statement, and here it is
exactly predictable.

``Delta_M(b) := LSE_m ll_m(b) - log M - ll(b, xi = 0)`` is EVEN in ``b``, to
floating point, for an antithetic member set.  Two facts compose:

  (i) the seam's ``logQ`` is invariant under ``(b, xi) -> (-b, -xi)``: the field
      term ``b u(xi)`` is manifestly so, and ``rho`` is
      ``log mean_p exp(b u_p(xi))``, likewise;
  (ii) PLAN sec 6.5 item 3's antithetic draws make the member set itself
      invariant under ``xi -> -xi``.

So ``Delta_M(-b) = Delta_M(b)`` EXACTLY, hence ``Delta_M(b) = c2 b^2 + c4 b^4 +
O(b^6)`` with no odd terms at all -- including no ``b^1`` term, which is what
would otherwise dominate: for a NON-antithetic member set the leading error is
``b * mean_m S1_m ~ b ||a||/sqrt(M)``, an ``O(b)`` Monte-Carlo error that swamps
the ``O(b^2)`` signal as ``b -> 0``.  Antithetic draws are therefore not a
variance-reduction nicety here; they are what makes P17 measurable at all.
Pinned by ``test_p17_estimator_is_even_in_b_gw``: measured
``Delta_M(-b) - Delta_M(+b)`` is EXACTLY 0.0 at both ``b = 0.1`` and
``b = 0.2``, bit for bit.

Measured residual against the exact finite-``M`` coefficient:

    b        |Delta_M(b) - c_M b^2|     ratio to previous (b^4 predicts 4.00)
    0.03536  1.247e-05
    0.05000  4.921e-05                  3.95
    0.07071  1.946e-04                  3.95
    0.10000  7.663e-04                  3.94
    0.14142  2.985e-03                  3.89
    0.20000  1.137e-02                  3.81
    0.28284  4.158e-02                  3.66      <- b^6 entering

    fitted log-log slope over b <= 0.14142 : 3.954   (predicted 4)
    fitted log-log slope over the full scan: 3.911

----------------------------------------------------------------------------
THE MONTE-CARLO ERROR -- asserted against, never a hand-picked tolerance
----------------------------------------------------------------------------

At finite ``M`` the estimator's ``b^2`` coefficient is not ``c_inf`` but the
EXACT finite-``M`` value obtained by expanding ``log mean_m exp(b S1_m + b^2
S2_m)``:

    S1_m = sum_i [ u_m(x_i) - <u_m(., z_i)>_F ]        (odd in xi)
    S2_m = -(1/2) sum_i Var_{p in F}( u_m(p, z_i) )    (even in xi)
    c_M  = mean_m S2_m + (1/2)[ mean_m S1_m^2 - (mean_m S1_m)^2 ]

with ``mean_m S1_m = 0`` identically under antithetic draws (measured: 0.0e+00).
Antithetic partners carry IDENTICAL ``S1^2`` and ``S2``, so the ``M = 512``
member set contributes exactly ``M/2 = 256`` independent values
``y_j = S2_j + S1_j^2 / 2`` and

    SE = std(y, ddof=1) / sqrt(M/2) = 0.367974        (12.1% of c_inf)

That is the irreducible precision of P17 at this ``M``; it falls only as
``sqrt(2/M)``, because the estimator's dominant fluctuation is that of
``S1^2``.  The gate is ``|D(b_min) - c_inf| <= 3 SE``.

Measured (``M = 512``, 256 antithetic pairs):

    c_M - c_inf         = +0.429752   =  1.17 SE      <- the MC error itself
    D(0.03536) - c_M    = -0.009976                   <- the b^2 truncation
    D(0.03536) - c_inf  = +0.419776   =  1.14 SE      <- THE P17 RESULT

----------------------------------------------------------------------------
THE CONFIGURATION, AND WHY EACH CHOICE IS LOAD-BEARING
----------------------------------------------------------------------------

Deliberately synthetic and small (18 catalog rows, 14 of them the fitted
footprint, ``M_sph x M_z = 10 x 4 = 40`` latent modes), so the dense basis
``Phi`` -- and hence ``K = Phi Phi^T`` -- is formed EXACTLY rather than
approximated.  ``Phi`` is separable, ``Phi_{(p,k),(s,t)} = phi_sph[p,s]
phi_z[k,t]``, which is precisely the object the seam consumes as
``row_fac_m[p] = phi_sph[p] @ Xi_m``, so ``K`` is the field's covariance by
construction and not a model of it.  Five choices make the closed form exact:

* **No observed hosts** (``ngals = 0``, ``wgals = 0``).  The aggregate
  completeness is then identically zero -- asserted, not assumed, in
  ``test_p17_configuration_premises`` -- so every event's prior mass sits in the
  missing branch the field modulates (``phi_i = 1`` in sec 6.5 eq. (6)) and the
  numerator is ``const + logQ`` with no ``logaddexp`` mixing.

* **Delta-function PE posteriors landing on ``zgrid`` nodes.**  Each event's
  samples are identical, and its ``dL`` is chosen so that ``z_of_dL`` returns
  ``zgrid[k]`` to 1e-18, hence ``_grid_bracket`` puts all the weight on ONE
  node and ``miss = b_lo * q_lo`` exactly.  Without this, ``_interp_row``
  interpolates ``base_miss * Q`` LINEARLY (not ``logQ``), which injects a
  spurious ``O(b^2)`` term -- small, but the same order as the entire signal.

  **The obvious construction does not work, and the reason is worth recording:
  ``z_of_dL(dL_of_z(z))`` is NOT the identity on ``redshift.grid.zgrid``.**
  ``z_of_dL`` interpolates onto ``cosmology.zgrid``, which has 500 log-spaced
  nodes over ``[0, zMax]`` against ``redshift.grid.zgrid``'s 1000 -- and 500
  ``linspace`` nodes are NOT a subset of 1000, since the spacings are
  ``log 6 / 499`` and ``log 6 / 999``.  Feeding ``dL_of_z(zgrid[k])`` back
  through ``z_of_dL`` therefore lands 3.2e-6 BELOW the node (6e-4 relative at
  ``z = 0.005``), enough to put the sample in cell ``k-1`` with ``t = 0.998``
  -- a two-node mixture, not a delta.  ``_dl_for_node`` inverts the
  interpolation on the cosmology grid explicitly instead, which is exact.  This
  is a property of the shipped cosmology helpers, not of the latent seam; it is
  benign for the production likelihood (the two grids differ by less than a
  ``zgrid`` cell everywhere) but it is fatal to a pin that needs ``t == 0``.

  Identical samples also zero the per-event PE Monte-Carlo variance, which is
  what lets the shipped selection guard pass at ``N_obs = 8`` with 800
  injections (it does not, with scattered samples -- measured ``-inf`` at 400,
  1500, 3000 and 6000 injections alike, because the guard's budget is spent on
  the PE variance, not on ``Neff``).

* **All injections OFF the fitted footprint.**  ``logQ`` is bit-zero there
  (pin P13b), so ``mu`` -- and with it the ``-N_obs <Phi>_sel`` subtraction of
  sec 6.5 eq. (6) -- is EXACTLY ``xi``-independent and drops out of the closed
  form.  This is arm (a)'s ``H = I`` condition realized physically rather than
  by switching the count channel off.

* **``z_depth = 0.05`` against a grid running to ``z = 5``.**  The field lives
  only below the depth, where a fraction ~1e-5 of each pixel's missing budget
  sits, so the CONDITIONAL (per-pixel) normalizer
  ``log Z_p = log int dz base_miss(p,z) Q_p(z)`` is ``xi``-dependent only at that
  order.  This is the production regime (PLAN PR-0: the field redistributes
  ~0.01% of the missing budget) and it is the ONLY approximation in the closed
  form.  It is measured, not assumed:
  ``test_p17_member_weight_is_the_closed_form_logq`` bounds the leakage at
  2.3e-5 RELATIVE, i.e. 3 orders of magnitude below ``SE / c_inf = 0.12``.

* **``b`` nodes on a SYMMETRIC Chebyshev-Lobatto grid** ``[-2, 2]`` (production
  uses ``[0, b_max]``), so the evenness pin can evaluate at ``-b`` without
  extrapolating ``interp_b``.  ``b = 0`` is node 16 exactly, which makes
  ``ll(b = 0)`` the exact ``logQ == 0`` reference rather than a limit.
  ``interp_b`` reproduces ``A_m(z; b) = sum_{p in F} e^{b u_m}`` to 8e-16
  relative over the scan, so the ``b``-interpolation contributes nothing.

Member spread, for the sec 6.5 record: ``sigma(ll_m) = b_GW * std(S1) =``
0.32 nats at ``b_GW = 0.1`` and 0.91 nats at ``b_GW = 0.283`` -- i.e. this
configuration reaches the top row of sec 6.5's ``M_draw`` table (``sigma = 1.0``
-> ``M_draw = 9`` for a 0.1-nat bias) while P17 still validates the estimator
there to 1.1 SE.

Runtime ~45 s on CPU, dominated by the single XLA compile of the ``M = 512``
member vmap; the seven-point ``b`` scan itself is 2.7 s.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import (
    C_MODE_AGGREGATE_STRUCT,
    CosmoParams,
    EMCatalog,
    GWEvent,
    SurveyParams,
)
from darksirens.gw.populations import get_fixed_population_params
from darksirens.likelihood.core import darksiren_log_likelihood
from darksirens.likelihood.latent_q import (
    footprint_row_map,
    interp_b,
    on_footprint_mask,
)
from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.redshift.latent_field import build_latent_basis, sky_moments
from darksirens.utils.cosmology import H0Planck, Om0Planck, dL_of_z, z_of_dL

NG = int(zgrid.size)
_Z = np.asarray(zgrid)

Z_DEPTH = 0.05
#: Below-depth grid nodes on THIS run's grid (see EV_NODE below).
_N_SUB = int((_Z <= Z_DEPTH).sum())
N_ROWS, N_FIT = 18, 14          # 14 footprint rows; rows 14..17 are outside F
M_SPH, M_Z = 10, 4              # 40 latent modes -> a dense Phi is 720 x 40
N_B, B_MAX = 33, 2.0            # symmetric Chebyshev-Lobatto b nodes on [-2, 2]
LS_SPH, LS_Z = 0.9, 0.06
N_EV, N_SAMP, N_SEL = 8, 4, 800
M_DRAW = 512                    # 256 ANTITHETIC pairs (PLAN sec 6.5 item 3)

#: The b_GW scan: geometric, ratio sqrt(2), so a b^4 residual must fall by 4.00
#: per step.  The top of the range is where b^6 starts to show (measured ratio
#: 3.66 in the last step), which is why the rate fit uses the small-b half.
B_SCAN = np.array([0.03536, 0.05, 0.07071, 0.1, 0.14142, 0.2, 0.28284])
_RATE_FIT_N = 5                 # points used for the log-log rate fit

COSMO = CosmoParams(H0=H0Planck, Om0=Om0Planck)
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))

#: ``b_miss`` IS ``b_GW`` in latent mode (PLAN sec 4.3 inverts the guard;
#: ``completion.latent_b_gw``).  ``z50``/``w`` put the survey's missing budget
#: out at z ~ 1, far above ``z_depth``, which is what makes the conditional
#: per-pixel normalizer's response to the field ~1e-5 (see the module docstring).
SURVEY = SurveyParams(
    n0=1.0, z50=1.0, w=0.30, delta=0.0, b_miss=0.0, alpha_miss=1.0,
    z_depth=Z_DEPTH, c_mode=C_MODE_AGGREGATE_STRUCT)


# --------------------------------------------------------------- the field
def _build():
    """Basis, antithetic draws, moments -- everything the seam consumes.

    The 8 event pixels are a tight cluster (0.10 rad scatter against
    ``ls_sph = 0.9``) and the remaining 6 footprint pixels are spread over the
    sphere.  Clustering is what makes the OFF-DIAGONAL kernel sum dominate: the
    closed form (T') cancels the diagonal against ``-V/2``, so uncorrelated
    events give ``c_inf ~ 0`` and P17 degenerates into testing that a small
    number is small.  With this geometry ``sum_ij Ktilde = 9.33`` against
    ``sum_i V = 3.25``, i.e. a 3.0 signal against a 0.37 MC error.
    """
    n_sub = int((_Z <= Z_DEPTH).sum())
    z_sub = _Z[:n_sub]

    rng = np.random.default_rng(17)
    vec = np.concatenate([
        np.array([0.0, 0.0, 1.0])[None, :] + 0.10 * rng.normal(size=(N_EV, 3)),
        rng.normal(size=(N_FIT - N_EV, 3)),
    ], axis=0)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)

    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_DEPTH, ls_sph=LS_SPH, ls_z=LS_Z)
    phi_sph = np.asarray(basis.phi_sph)          # (N_FIT, M_SPH)
    phi_z_sub = np.asarray(basis.phi_z_out)      # (n_sub, M_Z)

    # ANTITHETIC pairs: member m and member m + M/2 are xi and -xi.  This is
    # what makes Delta_M(b) exactly even in b (see the docstring), and it is
    # the configuration PLAN sec 6.5 item 3 specifies for the shipped ensemble.
    half = rng.normal(size=(M_DRAW // 2, M_SPH * M_Z))
    xi = np.concatenate([half, -half], axis=0)
    row_fac_fit = np.stack([phi_sph @ x.reshape(M_SPH, M_Z) for x in xi])

    f_p = rng.uniform(0.6, 1.0, size=N_FIT)
    b_nodes = B_MAX * (-np.cos(np.pi * np.arange(N_B) / (N_B - 1)))
    A_sub, B_sub = sky_moments(basis, xi, b_nodes, f_p, row_fac=row_fac_fit)

    A = np.zeros((M_DRAW, N_B, NG)); A[:, :, :n_sub] = np.asarray(A_sub)
    B = np.zeros((M_DRAW, N_B, NG)); B[:, :, :n_sub] = np.asarray(B_sub)
    phi_z = np.zeros((NG, M_Z)); phi_z[:n_sub] = phi_z_sub
    row_fac = np.concatenate(
        [row_fac_fit, np.zeros((M_DRAW, 1, M_Z))], axis=1)   # zero pad row

    return dict(n_sub=n_sub, phi_sph=phi_sph, phi_z_sub=phi_z_sub, xi=xi,
                row_fac_fit=row_fac_fit, row_fac=row_fac, phi_z=phi_z,
                A=A, B=B, b_nodes=b_nodes, f_p=f_p)


_F = _build()
_ROW_MAP = footprint_row_map(np.arange(N_ROWS), np.arange(N_FIT), N_FIT)
_ON_FP = np.asarray(on_footprint_mask(_ROW_MAP, N_FIT))

#: The field on the fitted footprint, (M_DRAW, N_FIT, n_sub) -- the SAME object
#: the seam reconstructs as ``row_fac_m[p] . phi_z[k]``, evaluated here in
#: numpy so the closed form is independent of the implementation under test.
_U = np.einsum('mpk,zk->mpz', _F["row_fac_fit"], _F["phi_z_sub"])

EV_PIX = np.arange(N_EV)
#: Event grid nodes, expressed as FRACTIONS of the below-depth block rather
#: than as absolute indices.  ``zgrid``'s size and spacing are set by
#: ``DARKSIRENS_ZMAX`` at import, so the original absolute ``[3, 6, ..., 26]``
#: was a statement about one grid only: at ``ZMAX = 6.0`` it has ``n_sub = 28``
#: nodes below the depth and those eight span z = 0.0054-0.0477, essentially the
#: whole block; at ``ZMAX = 1.0`` the same indices have ``n_sub = 71`` and span
#: only z = 0.0021-0.0182, the bottom quarter, packing every event into a few
#: correlation lengths.  That is a different physical configuration, and it
#: broke both the closed-form premise (3.4e-4 relative) and the b^4 convergence
#: rate (3.63 against the required 3.7) -- a real zmax sensitivity in the PIN,
#: not in the estimator it pins.  The fractions below reproduce the original
#: indices exactly at ZMAX = 6.0.
_EV_FRAC = np.array([3, 6, 9, 13, 17, 20, 23, 26]) / 27.0
EV_NODE = np.unique(np.rint(_EV_FRAC * (_N_SUB - 1)).astype(int))
assert EV_NODE.max() < _N_SUB, (
    f"event nodes {EV_NODE.tolist()} must sit below the depth "
    f"(n_sub = {_N_SUB} at DARKSIRENS_ZMAX = {zgrid[-1]:.3g})")


def _dl_for_node(k):
    """``dL`` whose ``z_of_dL`` is ``zgrid[k]`` -- exactly, not to 3e-6.

    ``z_of_dL`` is ``jnp.interp(dL, dL_of_z(cosmology.zgrid), cosmology.zgrid)``
    and ``cosmology.zgrid`` (500 nodes) is NOT a subset of
    ``redshift.grid.zgrid`` (1000), so ``dL_of_z(zgrid[k])`` does not round-trip
    to node ``k`` -- see the module docstring.  Invert the piecewise-linear map
    on the cosmology grid instead: the target ``z*`` sits at fraction ``s`` of
    cosmology cell ``j``, and the ``dL`` at the same fraction of that cell maps
    back to ``z*`` by construction.  Verified to 1e-18 in
    ``test_p17_configuration_premises``.
    """
    from darksirens.utils import cosmology as _cosmo

    zc = np.asarray(_cosmo.zgrid)
    dlc = np.asarray(dL_of_z(jnp.asarray(zc), H0Planck, Om0Planck))
    z_star = _Z[k]
    j = int(np.searchsorted(zc, z_star)) - 1
    s = (z_star - zc[j]) / (zc[j + 1] - zc[j])
    return float(dlc[j] + s * (dlc[j + 1] - dlc[j]))


_DL_EV = np.array([_dl_for_node(k) for k in EV_NODE])


def _latent_leaves(M=None):
    sl = slice(None) if M is None else slice(0, M)
    return dict(
        latent_row_fac=jnp.asarray(_F["row_fac"][sl]),
        latent_phi_z=jnp.asarray(_F["phi_z"]),
        latent_row_map=jnp.asarray(_ROW_MAP),
        latent_on_fp=jnp.asarray(_ON_FP),
        latent_A=jnp.asarray(_F["A"][sl]),
        latent_B=jnp.asarray(_F["B"][sl]),
        latent_b_nodes=jnp.asarray(_F["b_nodes"]),
        latent_P_F=float(N_FIT), latent_F_F=float(_F["f_p"].sum()),
    )


def _catalog(**extra):
    """An EMPTY catalog: no observed hosts anywhere.

    ``ngals = 0`` / ``wgals = 0`` makes ``dN_obs`` vanish, hence ``C(z) == 0``
    identically (asserted below) and the numerator's ``logaddexp(A_obs,
    log_miss)`` reduces to ``log_miss``.  That is Limit III's trigger stated
    literally -- "the catalogued fraction of each event's prior mass tends to
    zero" -- rather than approximated by a sparse catalog.
    """
    zg = jnp.full((N_ROWS, 1), 100.0)
    ng = jnp.zeros(N_ROWS, dtype=jnp.int32)
    kde, idx = build_pixel_kde_cache(
        np.arange(N_ROWS, dtype=np.int32), zg, N_ROWS, ngals=ng)
    fpr = np.zeros(N_ROWS); fpr[:N_FIT] = _F["f_p"]
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=jnp.ones((N_ROWS, 1)),
        wgals=jnp.zeros((N_ROWS, 1)), ngals=ng,
        delta_g_pix_z=jnp.zeros((N_ROWS, NG)), dN_obs_kde=kde,
        pixel_to_cache_idx=idx, unique_pixels=None,
        f_p_rows=jnp.asarray(fpr), **extra)


def _gw_pe():
    """Delta-function posteriors: every sample of an event is the SAME point.

    ``dL = dL_of_z(zgrid[k])`` makes ``z_of_dL`` land on node ``k`` exactly and
    ``_grid_bracket`` return ``t == 0``, so the seam's two-node gather collapses
    to one node and ``log numerator_i = const + logQ(p_i, k_i)`` EXACTLY.
    """
    m1 = np.repeat(np.linspace(30.0, 45.0, N_EV), N_SAMP)
    m2 = np.repeat(np.linspace(10.0, 16.0, N_EV), N_SAMP)
    n = N_EV * N_SAMP
    return GWEvent(
        m1det=jnp.asarray(m1), m2det=jnp.asarray(m2),
        dL=jnp.asarray(np.repeat(_DL_EV, N_SAMP)), chieff=jnp.zeros(n),
        prior_wt=jnp.ones(n),
        pixels=jnp.asarray(np.repeat(EV_PIX, N_SAMP).astype(np.int32)),
        q=jnp.asarray(m2 / m1), valid=jnp.ones(n, dtype=bool))


def _gw_sel():
    """Injections placed ENTIRELY outside the fitted footprint.

    ``logQ`` is bit-zero off ``F`` (pin P13b), so ``mu`` is exactly
    ``xi``-independent and eq. (6)'s ``- N_obs <Phi>_sel`` term is identically
    zero.  That is what reduces the shipped, selection-corrected likelihood to
    P17 arm (a)'s prior form without touching a flag.
    """
    r = np.random.default_rng(11)
    m1 = r.uniform(20, 60, N_SEL); m2 = r.uniform(8, 18, N_SEL)
    return GWEvent(
        m1det=jnp.asarray(m1), m2det=jnp.asarray(m2),
        dL=jnp.asarray(r.uniform(100.0, 2500.0, N_SEL)),
        chieff=jnp.zeros(N_SEL), prior_wt=jnp.ones(N_SEL),
        pixels=jnp.asarray(r.integers(N_FIT, N_ROWS, N_SEL).astype(np.int32)),
        q=jnp.asarray(m2 / m1), valid=jnp.ones(N_SEL, dtype=bool))


_GW_PE = _gw_pe()
_GW_SEL = _gw_sel()


def _ll(b_gw, *, M=None, cat=None):
    """The SHIPPED marginalized likelihood: ``logsumexp_m ll_m - log M``."""
    cat = _catalog(**_latent_leaves(M)) if cat is None else cat
    return float(darksiren_log_likelihood(
        COSMO, SURVEY._replace(b_miss=float(b_gw)), POP, _GW_PE, cat,
        _GW_SEL, cat, N_EV, N_SAMP, float(N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=True, lss_field_mode="latent"))


_CACHE: dict[float, float] = {}


def _delta(b_gw):
    """``Delta_M(b) = LSE_m ll_m - log M - ll(xi = 0)``.

    ``b_GW = 0`` is an exact Chebyshev-Lobatto node (``N_B`` odd on a symmetric
    interval), where ``A_m = P_F`` and ``B_m = F_F`` give ``rho == 0`` and the
    field term vanishes, so ``ll(0)`` IS the ``logQ == 0`` reference -- not an
    extrapolation towards one.
    """
    for b in (0.0, float(b_gw)):
        if b not in _CACHE:
            _CACHE[b] = _ll(b)
    return _CACHE[float(b_gw)] - _CACHE[0.0]


# ------------------------------------------------------ the closed form
def _closed_form():
    """The kernel contractions of (T'), and the exact finite-``M`` coefficient.

    ``K = Phi Phi^T`` is formed from the separable factors rather than the
    720 x 40 dense ``Phi``: ``Ktilde_ij = (s_i . s_j)(g_i . g_j)`` with
    ``s_i = phi_sph[p_i] - mean_F phi_sph`` and ``g_i = phi_z[k_i]``, and
    ``V(z_k) = Vsph ||g_k||^2`` with ``Vsph`` the footprint variance of
    ``phi_sph``.  Both are exactly the contractions written in the docstring;
    separability is a property of the basis the seam consumes, not a shortcut.
    """
    phi_sph, phi_z_sub = _F["phi_sph"], _F["phi_z_sub"]
    g = phi_z_sub[EV_NODE]                                  # (N_EV, M_Z)
    sph_mean = phi_sph.mean(axis=0)
    s = phi_sph[EV_PIX] - sph_mean[None, :]                 # (N_EV, M_SPH)
    k_tilde = (s @ s.T) * (g @ g.T)
    v_sph = (phi_sph ** 2).sum(axis=1).mean() - float(sph_mean @ sph_mean)
    v_i = v_sph * (g ** 2).sum(axis=1)
    c_inf = 0.5 * k_tilde.sum() - 0.5 * v_i.sum()

    # PLAN sec 1.6 / sec 6.3 as WRITTEN: the unprojected pairing, no normalizer.
    k_naive = (phi_sph[EV_PIX] @ phi_sph[EV_PIX].T) * (g @ g.T)
    c_naive = 0.5 * k_naive.sum()

    # Per-member expansion coefficients, exact (no b expansion of the code).
    u_ev = _U[:, EV_PIX, :][:, np.arange(N_EV), EV_NODE]        # (M, N_EV)
    u_bar = np.stack([_U[:, :, EV_NODE[i]].mean(axis=1)
                      for i in range(N_EV)], axis=1)
    u_var = np.stack([_U[:, :, EV_NODE[i]].var(axis=1)
                      for i in range(N_EV)], axis=1)
    s1 = (u_ev - u_bar).sum(axis=1)
    s2 = -0.5 * u_var.sum(axis=1)
    c_m = s2.mean() + 0.5 * (np.mean(s1 ** 2) - np.mean(s1) ** 2)

    # Antithetic partners share S1^2 and S2 exactly, so the independent unit is
    # the PAIR: M/2 values, not M.  Using M would understate SE by sqrt(2).
    n_pair = M_DRAW // 2
    y = (s2 + 0.5 * s1 ** 2)[:n_pair]
    se = float(y.std(ddof=1) / np.sqrt(n_pair))
    return dict(k_tilde=k_tilde, v_i=v_i, c_inf=float(c_inf),
                c_naive=float(c_naive), c_m=float(c_m), se=se, s1=s1, s2=s2)


_CF = _closed_form()


def _logq_closed_form(m, b):
    """Member ``m``'s ``logQ`` on ``(F, z_sub)``, from the definition.

    ``rho`` is rebuilt from ``log mean_{p in F} e^{b u}`` rather than from
    ``rho_from_moments``/``interp_b``, and the field from the numpy ``_U``, so
    this is an independent statement of what the seam is supposed to emit --
    a pin that reused the shipped helpers would only test that the code calls
    itself.
    """
    a_m = np.sum(np.exp(b * _U[m]), axis=0)                 # (n_sub,)
    return b * _U[m] - np.log(a_m / N_FIT)[None, :]


# ====================================================================== pins

def test_p17_configuration_premises():
    """Every premise the closed form rests on, asserted rather than assumed.

    A degenerate configuration must fail loudly here, not pass P17 vacuously.
    """
    from darksirens.redshift.completion import _latent_C_curve, _precompute_grids

    # (1) The catalog is empty, so C(z) == 0 identically: all of every event's
    #     prior mass is in the field-modulated missing branch.
    grids = _precompute_grids(
        COSMO, SURVEY._replace(b_miss=0.1), _catalog(**_latent_leaves(2)))
    c_curve = np.asarray(_latent_C_curve(grids))
    assert np.max(np.abs(c_curve)) == 0.0, (
        f"C(z) is not identically zero (max {np.max(np.abs(c_curve))}); the "
        "closed form's rho = log(A/P_F) branch and phi_i = 1 both fail")

    # (2) Each event lands on a zgrid node, so the seam's two-node gather puts
    #     essentially all of its weight on that ONE node.  The bracket may
    #     report (k-1, t = 1) rather than (k, t = 0) -- the sub-ulp residual of
    #     the inversion decides which side of the knot searchsorted picks -- so
    #     the assertion is on the WEIGHT the target node receives, which is the
    #     quantity the closed form actually needs.
    z_ev = np.asarray(z_of_dL(jnp.asarray(_DL_EV), H0Planck, Om0Planck))
    assert np.max(np.abs(z_ev - _Z[EV_NODE])) < 1e-15, (
        f"z_of_dL(_DL_EV) missed the grid nodes by "
        f"{np.max(np.abs(z_ev - _Z[EV_NODE])):.2e}; _interp_row would mix two "
        "nodes LINEARLY IN Q and inject a spurious O(b^2) term")
    idx = np.clip(np.searchsorted(_Z, z_ev, side="right") - 1, 0, NG - 2)
    t = (z_ev - _Z[idx]) / (_Z[idx + 1] - _Z[idx])
    w_target = np.where(idx == EV_NODE, 1.0 - t,
                        np.where(idx == EV_NODE - 1, t, 0.0))
    assert np.all(w_target >= 1.0 - 1e-12), (
        f"the gather is not concentrated on the event node: idx={idx} "
        f"(want {EV_NODE}) t={t} weight={w_target}")

    # (3) Every injection is off the fitted footprint, so mu is exactly
    #     xi-independent and eq. (6)'s -N_obs <Phi>_sel term vanishes.
    assert not np.any(_ON_FP[np.asarray(_GW_SEL.pixels)]), (
        "an injection landed inside F; mu would acquire an xi dependence and "
        "P17 would no longer be arm (a)")

    # (4) The event nodes are below the depth (where the field is nonzero) and
    #     the field is genuinely there.
    assert np.all(EV_NODE < _F["n_sub"])
    assert np.max(np.abs(_U)) > 0.1, "the latent field is numerically absent"

    # (5) b-interpolation is not a source of error: interp_b reproduces
    #     A_m(z; b) = sum_{p in F} e^{b u} to ~1e-15 across the scan.
    worst = 0.0
    for b in (B_SCAN[0], B_SCAN[-1]):
        direct = np.sum(np.exp(b * _U[0]), axis=0)
        got = np.asarray(interp_b(jnp.asarray(_F["A"][0]),
                                  jnp.asarray(_F["b_nodes"]), float(b)))
        worst = max(worst, float(np.max(np.abs(got[:_F["n_sub"]] / direct - 1))))
    assert worst < 1e-12, f"interp_b error {worst:.2e} pollutes the b^4 residual"

    # (6) Antithetic draws: the odd part of the member average is EXACTLY zero,
    #     which is what removes the O(b) Monte-Carlo term (see the docstring).
    s1 = _CF["s1"]
    assert abs(float(s1.mean())) < 1e-12 * float(s1.std()), (
        f"member draws are not antithetic: mean S1 = {s1.mean():.3e}")


def test_p17_member_weight_is_the_closed_form_logq():
    """Each member's likelihood IS ``ll(0) + sum_i logQ_m(x_i)``.

    The structural half of P17, and the measurement that bounds its ONLY
    approximation.  If this holds, the marginalization the estimator performs is
    exactly the Gaussian integral the closed form solves; the remaining question
    is whether the estimator performs it correctly, which the later pins ask.

    The residual is the response of the CONDITIONAL per-pixel normalizer
    ``log Z_p = log int dz base_miss(p, z) Q_p(z)``, which the field perturbs in
    proportion to the below-depth share of the pixel's missing budget.  Measured
    at <= 4e-5 relative, flat in ``b_GW`` (1.0e-05 / 8.9e-06 / 5.6e-06 for
    member 0 at b = 0.05 / 0.2 / 0.6) -- three orders of magnitude below the
    12% Monte-Carlo error the P17 gate is set at, and itself a measurement of
    PLAN PR-0's "the field redistributes ~0.01% of the missing budget".
    """
    ll0 = _ll(0.0, M=1)
    worst = 0.0
    for m in range(3):
        cat = _catalog(
            latent_row_fac=jnp.asarray(_F["row_fac"][m:m + 1]),
            latent_phi_z=jnp.asarray(_F["phi_z"]),
            latent_row_map=jnp.asarray(_ROW_MAP),
            latent_on_fp=jnp.asarray(_ON_FP),
            latent_A=jnp.asarray(_F["A"][m:m + 1]),
            latent_B=jnp.asarray(_F["B"][m:m + 1]),
            latent_b_nodes=jnp.asarray(_F["b_nodes"]),
            latent_P_F=float(N_FIT), latent_F_F=float(_F["f_p"].sum()))
        for b in (0.05, 0.2, 0.6):
            got = _ll(b, cat=cat) - ll0
            lq = _logq_closed_form(m, b)
            want = float(sum(lq[EV_PIX[i], EV_NODE[i]] for i in range(N_EV)))
            assert abs(want) > 1e-3, "member/b pair is vacuous (logQ ~ 0)"
            worst = max(worst, abs(got - want) / abs(want))
    # The tolerance is set from measurement on BOTH grids, because the effect it
    # bounds is physical and genuinely grid-dependent -- it is the below-depth
    # share of each pixel's missing budget, and ``DARKSIRENS_ZMAX`` sets the
    # grid's extent at import:
    #
    #   ZMAX = 6.0 (production): z_depth = 0.05 of 6   -> worst 4.0e-5
    #   ZMAX = 1.0 (fast suite): z_depth = 0.05 of 1   -> worst 2.9e-4
    #
    # A larger below-depth share means a larger normalizer response, so 2.9e-4
    # at ZMAX = 1.0 is the expected answer and not a failure. Gating at the
    # production number alone made this a silently zmax-sensitive pin. 1e-3
    # keeps ~3.4x headroom on the worse grid and is still two orders below the
    # 12% Monte-Carlo error the P17 gate itself is set at, so the premise this
    # guards -- "the per-pixel normalizer response is negligible against the
    # signal" -- is tested with room to spare on either grid.
    assert worst < 1e-3, (
        f"member log-weight departs from sum_i logQ(x_i) by {worst:.2e} "
        "relative; the closed form's premises (empty catalog, t = 0 gather, "
        "off-footprint injections, negligible per-pixel normalizer response) "
        "no longer hold and c_inf below is not the right target")


def test_p17_estimator_is_even_in_b_gw():
    """``Delta_M(-b) == Delta_M(+b)``: no odd term exists, at any order.

    The seam's ``logQ`` is invariant under ``(b, xi) -> (-b, -xi)`` and the
    antithetic member set is invariant under ``xi -> -xi``, so the estimator is
    an even function of ``b_GW`` to floating point.  Two consequences, both
    load-bearing for P17: the ``O(b)`` Monte-Carlo term that would otherwise
    dominate as ``b -> 0`` is identically absent, and the leading correction to
    the Gaussian limit is ``b^4``, not ``b^3``.  Measured:
    ``Delta_M(-b) == Delta_M(+b)`` bit for bit at ``b = 0.1`` and ``b = 0.2``.

    This is also a genuine seam pin: an implementation that clipped, floored or
    otherwise treated ``logQ`` asymmetrically -- or that lost the antithetic
    pairing -- would break it while every self-consistency pin still passed.
    """
    for b in (0.1, 0.2):
        plus, minus = _delta(b), _delta(-b)
        assert abs(plus) > 1e-4, "the b point is vacuous"
        assert abs(plus - minus) <= 1e-11 * abs(plus), (
            f"Delta_M is not even in b_GW at b = {b}: "
            f"{plus!r} vs {minus!r} (rel {abs(plus - minus) / abs(plus):.2e})")


def test_p17_gaussian_marginalization_limit():
    """**P17.** The estimator reproduces the closed form, within the MC error.

    ``D(b) = Delta_M(b) / b_GW^2`` must approach ``c_inf`` of (T'), which is
    built entirely from ``K = Phi Phi^T``.  The tolerance is NOT chosen: it is
    ``3 SE`` with ``SE`` the standard error of the ``b^2`` coefficient over the
    256 independent antithetic pairs, 0.367974 here, i.e. 12.1% of ``c_inf``.
    That precision falls only as ``sqrt(2 / M)`` -- P17 at ``M = 512`` cannot do
    better, and saying so is part of the result.

    Measured: ``D(0.03536) = 3.461972`` against ``c_inf = 3.042196``, a
    discrepancy of ``+0.419776 = 1.14 SE``, of which ``1.17 SE`` is the finite-
    ``M`` sampling error (``c_M - c_inf = +0.429752``) and ``-0.0100`` the
    ``b^2`` truncation.  The estimator agrees with truth.
    """
    c_inf, c_m, se = _CF["c_inf"], _CF["c_m"], _CF["se"]
    d = np.array([_delta(float(b)) for b in B_SCAN]) / B_SCAN ** 2

    # Non-vacuity: the field must genuinely move the likelihood, and the
    # closed-form target must be large compared with its own MC error.
    assert abs(_delta(float(B_SCAN[-1]))) > 0.1, (
        "the latent field moves the likelihood by < 0.1 nat at the top of the "
        "scan; P17 would be comparing two numbers that are both ~0")
    assert abs(c_inf) > 5.0 * se, (
        f"c_inf = {c_inf:.4f} is not resolved against SE = {se:.4f}; the event "
        "geometry has cancelled the off-diagonal kernel sum")

    d0 = float(d[0])
    assert abs(d0 - c_inf) <= 3.0 * se, (
        f"P17 FAILS: D(b={B_SCAN[0]}) = {d0:.6f} vs the closed form "
        f"{c_inf:.6f}; discrepancy {d0 - c_inf:+.6f} = "
        f"{(d0 - c_inf) / se:.2f} SE (SE = {se:.6f} over {M_DRAW // 2} "
        f"antithetic pairs). Finite-M coefficient c_M = {c_m:.6f}. "
        f"D over the scan: {np.array2string(d, precision=6)}")

    # The analytic finite-M coefficient must itself sit within the MC error of
    # truth -- otherwise the agreement above would be an accident of two errors.
    assert abs(c_m - c_inf) <= 3.0 * se, (
        f"the exact finite-M coefficient {c_m:.6f} is {abs(c_m - c_inf) / se:.2f} "
        f"SE from the closed form {c_inf:.6f}; the draws are not N(0, I)")


def test_p17_convergence_rate_is_b_to_the_fourth():
    """The agreement must improve at the PREDICTED rate, not merely improve.

    Since ``Delta_M`` is even in ``b_GW`` (pinned above), the residual against
    the exact finite-``M`` coefficient ``c_M`` -- an analytic number, not a fit,
    so this test has no free parameters -- must be ``O(b^4)``.  A wrong rate is
    how an estimator that is accidentally right at one amplitude is caught: an
    ``O(b^3)`` residual would mean the odd cumulants had not cancelled, and an
    ``O(b^2)`` residual would mean ``c_M`` itself is wrong, i.e. the closed form
    does not describe what the code computes.

    Measured slope 3.954 over ``b <= 0.14142``, with per-step ratios 3.945 /
    3.954 / 3.939 / 3.895 against the 4.00 a ``b^4`` correction requires (3.911
    over the full scan, whose last step is already 3.658 as ``b^6`` enters).
    """
    c_m = _CF["c_m"]
    b = B_SCAN[:_RATE_FIT_N]
    resid = np.abs(np.array([_delta(float(x)) for x in b]) - c_m * b ** 2)
    assert np.all(resid > 0.0)
    slope = float(np.polyfit(np.log(b), np.log(resid), 1)[0])
    assert 3.8 <= slope <= 4.2, (
        f"the residual against the closed form scales as b^{slope:.3f}, not "
        f"b^4; residuals {np.array2string(resid, precision=3)} at b = "
        f"{np.array2string(b)}")

    # Step-by-step, a sharper statement about SHAPE: each sqrt(2) step in b
    # shrinks the residual by 4.00, and a single bad step (a discontinuity in
    # the seam, say) would be invisible to a fit.  The band is wide on purpose
    # and the reason is worth stating, because a narrow one is not a stronger
    # test here -- it is a grid-dependent one.
    #
    # The residual at the smallest b is ~1e-5 nats, and it is a DIFFERENCE of
    # two O(1) quantities, so its own relative precision is the worst of the
    # scan.  That floor moves with the redshift grid, which ``DARKSIRENS_ZMAX``
    # sets at import.  Measured, same code, same configuration:
    #
    #   ZMAX = 6.0 (production, n_sub = 28):  3.945 / 3.954 / 3.939 / 3.895
    #   ZMAX = 1.0 (fast suite,  n_sub = 71):  3.579 / 3.748 / 3.830 / 3.839
    #
    # Note the ORDERING flips: at ZMAX = 6.0 the ratios decay from 3.95 as b^6
    # enters at the top of the scan; at ZMAX = 1.0 they RISE from 3.58, which is
    # the signature of noise at the bottom, not of a wrong power.  The fitted
    # slope -- gated above at [3.8, 4.2] and measured 3.954 / 3.911 on the two
    # grids -- is the robust statistic and is what discriminates b^4 from b^3 or
    # b^2.  This band only has to be tight enough to catch a step-change, so it
    # is set from the measured spread on BOTH grids with headroom, rather than
    # from the production grid alone (which is how it read as a zmax-sensitive
    # pin: it failed at ZMAX = 1.0 on a 3.58 that is noise).
    ratios = resid[1:] / resid[:-1]
    assert np.all(ratios > 3.4) and np.all(ratios < 4.2), (
        f"per-step residual ratios {np.array2string(ratios, precision=3)} are "
        "not the 4.00 a b^4 correction requires")


def test_p17_refutes_the_unprojected_kernel_form():
    """PLAN sec 1.6's stated target ``(b^2/2) sum_ij K(x_i, x_j)`` is REFUTED.

    Not a tolerance question: the seam's budget normalizer ``rho`` projects out
    the footprint monopole and cancels the events' self terms, and both effects
    enter at the SAME order ``b^2`` as the signal.  Measured here,
    ``c_naive = 26.931816`` against ``c_inf = 3.042196`` and a measured
    ``D = 3.461972`` -- the unprojected form is 8.9x too large and 63.8 SE away.

    Recorded as a pin because the same unprojected ``Phi_i`` appears in
    sec 6.5 eq. (6), whose ``sigma = ||L_H^{-1} a||_2`` is the plan's largest
    open number: the projection is a second systematic over-prediction of
    ``sigma``, on top of the Euclidean-vs-``H^{-1}`` one v4 already corrected,
    and both inflate the ``M_draw`` requirement exponentially.  Should sec 1.6
    ever be restated with the normalizer carried, this test's expectation --
    that ``c_naive`` is the WRONG answer -- is what must be revisited.
    """
    c_inf, c_naive, se = _CF["c_inf"], _CF["c_naive"], _CF["se"]
    d0 = _delta(float(B_SCAN[0])) / B_SCAN[0] ** 2

    assert abs(c_naive - c_inf) > 10.0 * se, (
        "the configuration cannot distinguish the projected from the "
        f"unprojected kernel sum ({c_naive:.4f} vs {c_inf:.4f}, SE {se:.4f}); "
        "this pin would be vacuous")
    assert abs(d0 - c_naive) > 10.0 * se, (
        f"the measured limit {d0:.6f} is CONSISTENT with PLAN sec 1.6's "
        f"unprojected form {c_naive:.6f} ({abs(d0 - c_naive) / se:.2f} SE). "
        "Either the seam stopped applying the budget normalizer or this "
        "module's derivation of (T') is wrong -- investigate before relaxing.")

    # And the seam-exact form is the one that fits.
    assert abs(d0 - c_inf) < abs(d0 - c_naive), (
        f"measured {d0:.6f} is closer to the unprojected form {c_naive:.6f} "
        f"than to the seam's closed form {c_inf:.6f}")


def test_p17_reports_the_member_spread():
    """The sec 6.5 cross-link: this configuration's member spread, on the record.

    ``sigma(ll_m) = b_GW * std_m(S1_m)`` exactly at leading order, so P17's own
    scan measures the quantity sec 6.5 item 5 predicts in closed form.  Measured
    ``std(S1) = 3.216484``, hence ``sigma = 0.32`` nats at ``b_GW = 0.1`` and
    ``0.91`` nats at ``b_GW = 0.28284`` -- the top of the scan sits on the first
    row of sec 6.5's table (``sigma = 1.0`` -> ``M_draw = 9`` for a 0.1-nat
    Jensen bias), and P17 validates the estimator there to 1.1 SE.

    The assertion is that the leading-order spread is the ACTUAL spread: the
    directly measured ``std_m(ll_m)`` (from the closed-form member weights,
    which the structure pin above ties to the shipped ones) must agree with
    ``b_GW std(S1)`` to better than 5% at the small-b end.  A configuration in
    which it did not would be one where the first-order expansion underlying
    both P17 and eq. (6) has already failed.
    """
    s1 = _CF["s1"]
    for b, tol in ((0.05, 0.05), (0.1, 0.05)):
        weights = np.array([
            float(sum(_logq_closed_form(m, b)[EV_PIX[i], EV_NODE[i]]
                      for i in range(N_EV)))
            for m in range(0, M_DRAW, 8)
        ])
        lead = b * float(s1[0:M_DRAW:8].std())
        assert lead > 0.05, "the member ensemble is degenerate"
        assert abs(weights.std() / lead - 1.0) < tol, (
            f"at b_GW = {b} the measured member spread {weights.std():.6f} "
            f"departs from the first-order prediction {lead:.6f} by more than "
            f"{100 * tol:.0f}%; eq. (6)'s linearization has failed here")
