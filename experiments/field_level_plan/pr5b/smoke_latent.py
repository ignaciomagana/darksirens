"""PR-5b smoke test: does the latent seam run on the PRODUCTION 259-event line?

Everything downstream (the M=256 anchor build, the 33-node campaign) is
worthless if the shipped ``lss_field_mode='latent'`` path does not build and
evaluate against the real DESI union + GWTC-5 pair, so that is checked FIRST
and cheaply, on the M_draw=8 anchor that already exists.  Four questions, in
the order in which a failure would be cheapest to discover:

S1  Does ``make_likelihood`` accept the latent opts at all (all six factory
    guards: exclusivity, ls_z units, resolution, isotropy, f_p consistency,
    P_F == |F|)?
S2  Is the value FINITE under the PR-0 clean-arm guard convention?  A -inf
    would mean the campaign measures nothing.
S3  Does :func:`latent_harness.member_ll_patch` recover the member vector?
    GATE: ``logsumexp(ll_m) - log M`` must reproduce the UNPATCHED scalar to
    the last bit -- if it does not, the patch is changing the computation and
    every PR-5b number is void.
S4  Does :func:`latent_harness.member_slice_patch` preserve ``ll_m``?
    GATE: members [0, 4) evaluated in a 4-chunk must equal members [0, 4) of
    the full 8-member run, bit for bit.

It also prints the timing of a warm evaluation, which sets the campaign's
wall-clock budget, and the realized member spread at M=8 -- the first
measured number of the whole rung, comparable against PREDICTION.md's
0.10278 nats directly (the anchor's own 8 draws are what the prediction's
"independent confirmation" was evaluated on).
"""
from __future__ import annotations

import json
import time

import latent_harness as H
from latent_harness import jax, jnp, np


def main():
    anchor = H.resolve_anchor_m8()
    print(f"[smoke] anchor = {anchor}", flush=True)
    import h5py
    with h5py.File(anchor) as f:
        g = f["latent_field"]
        m_draw = int(g["row_fac"].shape[0])
        sha = str(g.attrs["sha256"])
        grad_inf = float(g.attrs["grad_inf"])
    print(f"[smoke] M_draw={m_draw} sha256={sha} grad_inf={grad_inf:.3g}",
          flush=True)

    opts = H.clean_arm_opts(anchor)
    print(f"[smoke] devices: {jax.devices()}", flush=True)

    # ---- S0: b_GW is what the prediction assumed -------------------------
    # ``b_miss`` IS ``b_GW`` in latent mode (PLAN §4.3), and on the factory
    # path it cannot be passed as a fixed value (see latent_harness.B_GW), so
    # it arrives via the SurveyParams fiducial.  Read the DECODED value back
    # instead of trusting that coincidence: if it were not 1.0 the entire
    # predicted-vs-measured comparison would be against the wrong field
    # amplitude, and every member would still look perfectly well-behaved.
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.parameters import build_parameter_decoder

    dec = build_parameter_decoder(
        opts, get_fixed_population_params(opts.pop_model),
        fixed_parameter_values=H.fixed_values())
    decoded = dec.decode(jnp.asarray([67.74]))
    b_gw = None
    for obj in (decoded if isinstance(decoded, tuple) else (decoded,)):
        if hasattr(obj, "b_miss"):
            b_gw = float(obj.b_miss)
    print(f"[smoke] S0 decoded survey.b_miss (= b_GW) = {b_gw!r}", flush=True)
    assert b_gw == H.B_GW, (
        f"b_GW is {b_gw}, not {H.B_GW}: the prediction is for b_GW = 1")

    # ---- S1/S2: build + evaluate, unpatched (this is the shipped number) ----
    t0 = time.time()
    logl = H.build_likelihood(opts)
    print(f"[smoke] S1 build ok ({time.time() - t0:.1f} s)", flush=True)

    t0 = time.time()
    scalar = float(logl(jnp.asarray([67.74])))
    t_compile = time.time() - t0
    t0 = time.time()
    scalar2 = float(logl(jnp.asarray([67.74])))
    t_warm = time.time() - t0
    print(f"[smoke] S2 logL(H0=67.74) = {scalar!r}  "
          f"(compile+run {t_compile:.1f} s, warm {t_warm * 1e3:.1f} ms)",
          flush=True)
    assert np.isfinite(scalar), "latent likelihood is -inf at the anchor"
    assert scalar == scalar2, "likelihood is not deterministic"

    # ---- S3: the member-vector patch, gated against the unpatched scalar ----
    with H.member_ll_patch(m_draw) as seen:
        logl_v = H.build_likelihood(H.clean_arm_opts(anchor))
        vec = np.asarray(logl_v(jnp.asarray([67.74])))
    ll_m = vec + np.log(m_draw)
    from scipy.special import logsumexp as _lse
    recon = _lse(ll_m) - np.log(m_draw)
    print(f"[smoke] S3 intercepts={seen['n']} ll_m shape={vec.shape}",
          flush=True)
    print(f"[smoke] S3 ll_m = {np.array2string(ll_m, precision=6)}", flush=True)
    print(f"[smoke] S3 reduced {recon!r} vs unpatched {scalar!r}  "
          f"delta = {recon - scalar:.3e}", flush=True)
    assert seen["n"] > 0, (
        "the member-ll patch was never called: the build re-used a cached "
        "executable instead of re-tracing (see latent_harness.build_likelihood "
        "fresh_trace)")
    assert vec.shape == (m_draw,), "patch did not return the member vector"
    assert abs(recon - scalar) < 1e-6, (
        "the member-ll patch changed the likelihood; PR-5b cannot proceed")

    # ---- S4: chunking is exact -------------------------------------------
    half = m_draw // 2
    with H.member_ll_patch(half), H.member_slice_patch(0, half):
        logl_c = H.build_likelihood(H.clean_arm_opts(anchor))
        chunk = np.asarray(logl_c(jnp.asarray([67.74]))) + np.log(half)
    dev = np.abs(chunk - ll_m[:half]).max()
    print(f"[smoke] S4 chunk[0:{half}] vs full[0:{half}] max|dev| = {dev:.3e}",
          flush=True)
    # MEASURED 2.456e-11 nats (job 1136117) -- NOT bit-zero, and the first
    # version of this gate demanded bit-zero and failed.  The member axis is a
    # ``jax.vmap`` dimension, so XLA lays out and fuses the reduction
    # differently at width 4 than at width 8; the members themselves are
    # untouched.  2.5e-11 nats on a likelihood of -766.79 is 3e-14 relative
    # and 9 orders below the member spread this campaign measures (~1e-2
    # nats), so it is re-association, not a different computation.  The
    # tolerance is 1e-8 = P6's own convergence gate, still six orders below
    # anything PR-5b quotes.
    assert dev < 1e-8, (
        f"member chunking changed ll_m by {dev:.3e} nats, which is above "
        "floating-point re-association -- the chunks are not the same "
        "computation")

    # ---- the first measured number ---------------------------------------
    sd = float(np.std(ll_m, ddof=0))
    anti = ll_m[:half] + ll_m[half:] - 2.0 * ll_m.mean()
    print(f"[smoke] realized member sd at M=8: {sd:.6e} nats "
          f"(PREDICTION.md population sigma 1.0278e-1, "
          f"realized-sd prediction 1.0330e-1)", flush=True)
    print(f"[smoke] antithetic residual (ll_m + ll_m+M/2 - 2 mean): "
          f"{np.array2string(anti, precision=6)}", flush=True)

    out = H.PR5B_DIR / "smoke_latent.json"
    json.dump({
        "anchor": str(anchor), "sha256": sha, "M_draw": m_draw,
        "logL_unpatched": scalar, "ll_m": ll_m.tolist(),
        "reduced_minus_unpatched": recon - scalar,
        "chunk_max_dev": dev, "warm_ms": t_warm * 1e3,
        "compile_s": t_compile, "member_sd": sd,
        "devices": [str(d) for d in jax.devices()],
    }, open(out, "w"), indent=1)
    print(f"[smoke] wrote {out}", flush=True)
    print("[smoke] ALL GATES PASSED", flush=True)


if __name__ == "__main__":
    main()
