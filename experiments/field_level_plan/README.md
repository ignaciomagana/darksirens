# Field-level dark sirens — proposal bundle

Upload this zip to Overleaf via **New Project → Upload Project**.

## What is here

| file | role | compiles to |
|---|---|---|
| `proposal.tex` | **the main document** — set this as the Overleaf main file | 31 pp |
| `MODEL.tex` | the mathematical body. A LaTeX **body fragment**: no `\documentclass`, by design, because it was written to be dropped into a document | — |
| `MODEL_standalone.tex` | a thin wrapper that compiles `MODEL.tex` on its own, with exactly the preamble its header asks for | 17 pp |

Both documents build with `pdflatex` (run twice for cross-references) and need
only packages in TeX Live base: `amsmath`, `amssymb`, `amsthm`, `booktabs`,
`array`, `longtable`, `geometry`, `enumitem`, `algorithm`, `algorithmic`,
`xcolor`, `verbatim`, `hyperref`. No figures, no bibliography files, no custom
classes — nothing to upload alongside.

**Set `proposal.tex` as the main document** in Overleaf (Menu → Main document);
Overleaf may otherwise pick `MODEL.tex`, which cannot compile alone.

## Status of the numbers

Every quantitative claim in `proposal.tex` was audited against the measured
record of the implementation ladder (rungs PR-0 through PR-8) and **30
contradicted numbers were corrected**. Where the document states a measurement
it now cites the rung that produced it; where it states a projection it says so.

Points a reader should not miss, because they are the ones that changed:

* The production likelihood baseline is **1417 ms** (H100). The previously
  quoted 3027 ms is an **A100-80** number, so cost percentages deflate 52x,
  not 110x.
* The member-spread closed form was wrong by **171x** — the error was the
  Hessian, not the vector, and the sign of its stated caveat was backwards.
  Measured: 1.15 nats at H0 = 20, 1.49e-2 at the anchor.
* The stated Gaussian-marginalisation limit was wrong by **8.9x**: it dropped
  the budget normaliser, which enters at the same order. The unification claim
  survives — every term is still a kernel contraction — but the naive pairing
  does not.
* The mock closure tiers B and C **fail**, and the document says so. A
  variance decomposition localises 82–92% of the scatter to the event draw at
  fixed catalog, with the no-field control overconfident by 2.25x, so those
  tiers measure their own PE calibration rather than the field.

## Provenance

Generated from `experiments/field_level_plan/` in the `darksirens` repository.
The per-rung reports (`pr0/`, `pr3/`, `pr5b/`, `pr6a/`, `pr7/`, `pr8/`) hold the
raw measurements every corrected number traces to.
