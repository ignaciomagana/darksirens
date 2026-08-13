# Member-marginalization profile (backend=cpu)
# nEvents=8 nsamp=256 N_sel=800 N_rows=16 gals/row=8 sel_batch_size=None repeats=5

| K | M | value ref (ms) | value fac (ms) | value speedup | grad ref (ms) | grad fac (ms) | grad speedup |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 8.26 | 8.58 | 0.96x | 48.88 | 49.47 | 0.99x |
| 1 | 8 | 11.98 | 11.98 | 1.00x | 54.76 | 51.42 | 1.07x |
| 1 | 32 | 29.90 | 22.83 | 1.31x | 60.57 | 62.12 | 0.98x |
| 1 | 64 | 37.44 | 35.75 | 1.05x | 79.13 | 79.77 | 0.99x |

Value path uses the redshift-prior optimization barrier (materialize=True, dynesty value path); grad path uses materialize=False (numpyro NUTS path).  speedup = reference / factored.
