"""Is the event sum ADDITIVE across disjoint event sets?  The decisive test.

Every previous probe of the coupling had a flaw.  The invariance check needed the
capture's per-event ORDER, which is data-dependent.  The LOO probe needed the
selection correction to cancel, and it does not (it is non-linear in ``N``).
This one needs neither.

Define, for an event set ``X``,

    E(X) = Sum_i ll_i  =  (the captured per-event log-evidences, SUMMED)

The sum is invariant to whatever order the reduction ran in -- ``capture_check.py``
verified the captured multiset is block-invariant, and a sum does not care about
order at all.  ``E`` is also exactly the total minus the selection correction, so
it isolates the event side from the term that is known not to decompose.

Then for DISJOINT sets ``A`` and ``B``:

    E(A) + E(B) == E(A u B)      if and only if the per-event terms are additive
                                 and each depends on its own event alone.

The residual ``E(A u B) - E(A) - E(B)`` IS the coupling, in nats, with no
modelling assumption and no ordering assumption.  Run it with the real ``f_p``,
with no ``f_p``, and with ``f_p == 1`` everywhere:

* residual 0 without ``f_p``, non-zero with it  -> confirms the localisation;
* residual 0 at ``f_p == 1``                    -> the coupling needs f_p's
  VALUES, i.e. something that ``f_p`` makes non-uniform across rows;
* residual non-zero at ``f_p == 1``             -> the coupling is in the ``f_p``
  CODE PATH itself, since multiplying by exactly 1.0 is the identity in IEEE and
  cannot change a correctly-additive answer.

``f_p == 1`` is the sharp fork, and it is the reason this test is worth running
before any float-precision hypothesis: at ``f_p == 1`` the arithmetic is
mathematically identical to the no-``f_p`` arm, so any residual is structural.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from event_reshuffle import build_regrouped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--pattern", default="c00_e{:03d}")
    ap.add_argument("--n-sources", type=int, default=16)
    ap.add_argument("--n-a", type=int, default=12, help="events in set A")
    ap.add_argument("--n-b", type=int, default=12, help="events in set B")
    ap.add_argument("--nobs", type=int, default=60)
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--unit-fp", action="store_true",
                    help="replace the depth map with f_p == 1 everywhere")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--dh", type=float, default=2.0)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import arms as A
    import tier_b
    import darksirens.likelihood.core as _core

    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    sources = [Path(a.tree) / a.pattern.format(i) for i in range(a.n_sources)]

    if a.unit_fp:
        # A unit depth map: masked_frac == 0 everywhere, so f_p == 1 exactly.
        # Same nside and ordering as the real one, so nothing else changes.
        src_map = tier_b.paths_for(sources[0])["mth"]
        unit = work / "mth_unit.h5"
        with h5py.File(src_map) as f, h5py.File(unit, "w") as g:
            for k, v in f.attrs.items():
                g.attrs[k] = v
            for k in f:
                arr = np.asarray(f[k][...])
                if k == "masked_frac":
                    arr = np.zeros_like(arr)
                g.create_dataset(k, data=arr)
        real_opts = A.make_opts

        def _opts(paths, arm, **kw):
            kw.setdefault("mth_override", str(unit))
            return real_opts(paths, arm, **kw)

        A.make_opts = _opts
        print(f"[additivity] using a UNIT f_p map: {unit}", flush=True)

    ev_store = []
    real_ev = _core.log_evidence_and_mc_variance

    def _spy(ldw, nsamp):
        ll, var = real_ev(ldw, nsamp)
        jax.debug.callback(
            lambda x: ev_store.append(np.asarray(x).ravel().copy()), ll)
        return ll, var

    hs = [a.h0 - a.dh, a.h0, a.h0 + a.dh]

    def E_of(picks, tag):
        """Sum of per-event log-evidences at each H0 node, order-free."""
        d = build_regrouped(sources, picks, work / tag)
        p = tier_b.paths_for(d)
        logl, opts, data = A.build(p, a.arm)
        out_E, out_L = [], []
        for x in hs:
            ev_store.clear()
            v = float(logl(jnp.asarray([float(x)])))
            ev = np.concatenate(ev_store) if ev_store else np.array([])
            if ev.size != len(picks):
                raise SystemExit(f"{tag}: captured {ev.size} != {len(picks)}")
            out_E.append(float(ev.sum()))
            out_L.append(v)
        return np.array(out_E), np.array(out_L)

    # A and B disjoint: A from source 0, B from source 1.
    setA = [(0, e) for e in range(a.n_a)]
    setB = [(1, e) for e in range(a.n_b)]
    t0 = time.time()
    _core.log_evidence_and_mc_variance = _spy
    try:
        EA, LA = E_of(setA, "A")
        print(f"[additivity] E(A) = {EA[1]:.9f}  ({time.time()-t0:.0f}s)",
              flush=True)
        EB, LB = E_of(setB, "B")
        print(f"[additivity] E(B) = {EB[1]:.9f}", flush=True)
        EAB, LAB = E_of(setA + setB, "AB")
        print(f"[additivity] E(AuB) = {EAB[1]:.9f}", flush=True)
    finally:
        _core.log_evidence_and_mc_variance = real_ev

    resid = EAB - EA - EB
    # the same residual on the SCORE, which is what J is built from
    dE = lambda E: (E[2] - E[0]) / (2 * a.dh)  # noqa: E731
    resid_score = dE(EAB) - dE(EA) - dE(EB)
    scale = float(np.abs(EAB[1]))
    print(f"\nADDITIVITY (arm={a.arm}, unit_fp={a.unit_fp}, "
          f"|A|={a.n_a}, |B|={a.n_b}):")
    print(f"  E(AuB) - E(A) - E(B) = {resid[1]:+.6e} nat   "
          f"(relative to |E| = {scale:.3f}: {resid[1]/max(scale,1e-30):+.3e})")
    print(f"  the same on the SCORE dE/dH0 = {resid_score:+.6e}")
    print("  0 => the event sum is additive and each term depends on its own "
          "event alone.")
    json.dump(dict(arm=a.arm, unit_fp=bool(a.unit_fp), h0=a.h0, dh=a.dh,
                   n_a=a.n_a, n_b=a.n_b,
                   E_A=EA.tolist(), E_B=EB.tolist(), E_AB=EAB.tolist(),
                   logL_A=LA.tolist(), logL_B=LB.tolist(), logL_AB=LAB.tolist(),
                   residual=resid.tolist(), residual_at_peak=float(resid[1]),
                   residual_score=float(resid_score),
                   residual_relative=float(resid[1] / max(scale, 1e-30))),
              open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("ADDITIVITY_DONE", flush=True)


if __name__ == "__main__":
    main()
