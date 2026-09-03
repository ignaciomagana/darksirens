# Analyzing runs

`darksirens_analyze` post-processes one or more finished run directories: it
recomputes posterior-predictive population distributions, plots the cosmology
posteriors, and compares models by evidence.

```bash
darksirens_analyze \
  --run_dirs runs/compare/*/ \
  --nm 128 \
  --nq 48 \
  --nz 32 \
  --outdir figures/compare
```

## Inputs

`--run_dirs` takes any number of directories written by `darksirens_inference`
(default: the current directory). Each is read as `results.hdf5` plus
`settings.json`, with missing `settings.json` entries backfilled from the HDF5
attributes. When `results.hdf5` is absent or its write did not finish, the
analyzer falls back to the `samples.npy` crash-recovery chain and takes metadata
from `settings.json`; that path carries no evidence, so the model comparison is
skipped. `--allow_legacy_pickle` additionally permits a very old pickled-dict
`samples.npy`, which executes arbitrary Python from the file and should be used
only on files you trust.

## What it reports

For every run the analyzer draws population parameters from the posterior,
evaluates the population density on a shared grid, and summarises the median and
the `--cred_lo`/`--cred_hi` percentiles (default 5 and 95).

| Output | Content |
| --- | --- |
| `pm1_all_models.pdf`, `pm2_all_models.pdf`, `pq_all_models.pdf`, `pz_all_models.pdf`, `pchi_all_models.pdf` | 1-D posterior-predictive spectra, all runs overlaid. |
| `pm1m2_all_models.pdf` | Median 2-D joint $p(m_1, m_2)$ per run. |
| `cosmology_posterior.pdf` | `H0`, `Om0`, `w0`, `wa` posteriors with the injected value marked; skipped when cosmology was fixed in every run. |
| `rate_dNdz.pdf` | Normalised $dN/dz$, built with each run's full sampled or fixed cosmology. |
| `model_evidences.pdf`, `bayes_factors.pdf` | Relative $\log_{10} Z$ per run and the pairwise Bayes-factor matrix. |
| `latents_<tag>.pdf` | Latent caterpillar for GP and binned-GP runs; `<tag>` is the run directory name. |
| `sky_dipole_<tag>.pdf`, `sky_gp_map_<tag>.pdf`, `sky_multipole_cl_<tag>.pdf` | Sky-anisotropy summaries for a run with an anisotropic `--sky_model`. |
| `catalog_weights_<tag>.pdf`, `catalog_weights_<tag>.npy` | Derived catalog host fractions $w_1 \dots w_K$ for a K-catalog run. |

Everything lands in `--outdir` (default the current directory, created if
needed). The evidence table and the pairwise Bayes factors are also printed to
the terminal, one row per run, in $\log_{10} Z$.

The grid is set by `--mmin` (1.0), `--mmax` (100.0) and `--nm` (128) for primary
mass, `--nq` (48) over $q \in [0.01, 1]$, `--nz` (32) over $[0,$ `--zmax` $]$
(default 2.0), and `--nchi` (24) from `--chimin` to `--chimax` (default -1 to 1).
`--overlay_events` adds the observed detector-frame $m_1$ medians to $p(m_1)$,
and `--sky_nside` (16) sets the resolution of the `sphere_gp` posterior sky map.

Memory is sized automatically: `--batch_size` (posterior samples per batch) and
`--grid_chunk` (rows of the flattened $(m_1, q)$ plane per density slab) are
probed from the device when unset, under a `--max_mem_gb` budget (or
`DARKSIRENS_ANALYZE_MAX_MEM_GB`) times `--mem_safe_frac` (0.4). Lower `--nm`,
`--nq`, `--nz` and `--nchi` for a quick pass; see [Performance](performance.md).

## Reading a run directly

`results.hdf5` and `settings.json` are plain formats, so a run can be
post-processed without the analyzer:

```python
import json, h5py

run = "runs/compare/powerlaw+peak__spectral_sirens__dynesty__seed22__2026-01-01T00-00-00"
with h5py.File(f"{run}/results.hdf5", "r") as f:
    samples = f["samples"][()]                          # (N_samples, N_dim)
    labels = [s.decode() for s in f["labels"][()]]      # aligned with the columns
    logZ, logZerr = f.attrs["logZ"], f.attrs["logZerr"]
    complete = bool(f.attrs["result_complete"])
    fixed = dict(zip([s.decode() for s in f["fixed_labels"][()]],
                     f["fixed_values"][()])) if "fixed_labels" in f else {}
settings = json.load(open(f"{run}/settings.json"))

h0 = samples[:, labels.index("H0")]
print(settings["pop_model"], settings["universe_model"], complete, logZ, fixed)
```

`labels`, `lower_bound` and `upper_bound` are all aligned with the columns of
`samples`. `log_weights` and `log_likelihood` are per posterior sample when
present, whereas `logl_dead` and `logwt_dead` are indexed by retired live point
(length `n_dead`) and must not be zipped with `samples`. `logZ` is the raw
sampler value; `logZ_corrected` subtracts `log_prior_volume_fraction` for sky
models that reject part of their prior box, and is the comparable number between
two such models.
