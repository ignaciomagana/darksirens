from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import h5py
import healpy as hp
from tqdm import tqdm
import matplotlib.pyplot as plt

from darksirens.cli.common import _banner, _section, _row, _end, _ok, _fatal

#: Optional per-galaxy "mark" columns -> EMCatalog mark dataset name.  Read from
#: the raw catalog if present (for the marked-host model); padded like zgals.
MARK_INPUT_COLUMNS = {
    "mark_logmstar": "LOGMSTAR",
    "mark_logssfr": "LOGSSFR",
    "mark_metallicity": "LOGZ",
    "mark_color": "GR_COLOR",
}

def plot_diagnostics(counts, zs, npix, save_dir, nside):
    """Generates and saves diagnostic plots for the survey data."""
    _section("Diagnostic plots")

    # 1. Skymap of Galaxies
    plt.figure(figsize=(10, 6))
    # Replace 0s with NaN so empty pixels show up as gray instead of the lowest color map value
    map_to_plot = np.where(counts > 0, counts, np.nan)
    hp.mollview(map_to_plot, title=f"Galaxy Density Skymap (Nside={nside})", cmap='viridis', return_projected_map=False)
    plt.savefig(save_dir / f'skymap_density_nside_{nside}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Redshift Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(zs, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title("Survey Redshift Distribution")
    plt.xlabel("Redshift (z)")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    # Redshift is independent of nside, but we tag the filename to keep batch runs organized
    plt.savefig(save_dir / f'redshift_distribution_nside_{nside}.png', dpi=300)
    plt.close()

    # 3. Galaxies per Pixel Distribution
    plt.figure(figsize=(8, 5))
    valid_counts = counts[counts > 0]
    plt.hist(valid_counts, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
    plt.title(f"Galaxies per Pixel Distribution (Nside={nside}, Non-empty pixels)")
    plt.xlabel("Number of Galaxies")
    plt.ylabel("Frequency")
    plt.yscale('log') # Log scale is usually better for density distributions
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_dir / f'pixel_occupancy_distribution_nside_{nside}.png', dpi=300)
    plt.close()

    _ok(f"skymap_density_nside_{nside}.png")
    _ok(f"redshift_distribution_nside_{nside}.png")
    _ok(f"pixel_occupancy_distribution_nside_{nside}.png")
    _row("Directory", save_dir)
    _end()

def main(argv=None):
    optp = ArgumentParser(description="Process galaxy survey data into HEALPix pixels.")
    optp.add_argument("--survey_path", required=True, help="Path to the input HDF5 survey data")
    optp.add_argument("--save_path", default='./', help="Directory to save the outputs")
    optp.add_argument("--nside", type=int, default=64, help="HEALPix Nside parameter")
    optp.add_argument("--add_plots", action="store_true", help="Generate diagnostic plots")
    optp.add_argument(
        "--z_depth", type=float, default=None,
        help=(
            "Optional survey redshift depth: the redshift beyond which this "
            "survey does not attempt to detect galaxies. Written as "
            "f.attrs['z_depth'] and read by darksirens_inference to bound the "
            "completion missing-galaxy budget to z <= z_depth instead of the "
            "full [0, DARKSIRENS_ZMAX] grid. Omit for the legacy behaviour "
            "(no attribute written; the full-grid budget is used)."
        ),
    )

    opts = optp.parse_args(argv)

    survey_file = Path(opts.survey_path)
    save_dir = Path(opts.save_path)
    nside = opts.nside

    print()
    _banner("DARK SIRENS PIXELATE")
    print()

    # 1. Validation
    if not survey_file.is_file():
        _fatal(f"Survey file not found at {survey_file}")

    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    _section("Loading survey")
    _row("Input", survey_file)
    with h5py.File(survey_file, 'r') as f:
        ras = np.array(f['TARGET_RA']) * np.pi / 180
        decs = np.array(f['TARGET_DEC']) * np.pi / 180
        zs = np.array(f['Z'])
        ddzs = np.array(f['ZERR'])
        wts = np.array(f['WEIGHT'])
        marks_in = {
            ds: np.array(f[col], dtype=float)
            for ds, col in MARK_INPUT_COLUMNS.items() if col in f
        }
    if marks_in:
        _ok(f"Mark columns:           {', '.join(sorted(marks_in))}")

    ngals_total = len(ras)
    _ok(f"Galaxies loaded:        {ngals_total:,}")
    _end()

    # 3. Calculate HEALPix Indices
    _section("Pixelating")
    npix = hp.nside2npix(nside)
    _row("NSIDE", nside)
    _row("Pixels", f"{npix:,}")
    ind = hp.ang2pix(nside, np.pi/2 - decs, ras)

    # Calculate counts per pixel instantly
    counts = np.bincount(ind, minlength=npix)
    maxgals = counts.max()
    _ok(f"Max galaxies/pixel:     {maxgals:,}")

    # 4. Pre-allocate dense arrays with padding values
    # Padding defaults: z=100, dz=1, w=0
    cats_out = np.full((npix, maxgals), 100.0, dtype=zs.dtype)
    dzcats_out = np.full((npix, maxgals), 1.0, dtype=ddzs.dtype)
    dwcats_out = np.full((npix, maxgals), 0.0, dtype=wts.dtype)
    # Marks padded with 0 (masked downstream by ngals; centred at load).
    marks_out = {ds: np.full((npix, maxgals), 0.0, dtype=float) for ds in marks_in}

    # 5. Fast Vectorized Grouping
    sort_idx = np.argsort(ind)
    sorted_ind = ind[sort_idx]

    # Sort the data arrays based on pixel index
    sorted_zs = zs[sort_idx]
    sorted_ddzs = ddzs[sort_idx]
    sorted_wts = wts[sort_idx]
    sorted_marks = {ds: arr[sort_idx] for ds, arr in marks_in.items()}

    # Find the boundary indices for each unique pixel
    unique_pix, start_indices = np.unique(sorted_ind, return_index=True)

    # Fill the pre-allocated arrays
    for i, pix in enumerate(tqdm(unique_pix, desc="Populating dense arrays")):
        start = start_indices[i]
        end = start_indices[i+1] if i + 1 < len(start_indices) else len(sorted_ind)
        count = end - start

        cats_out[pix, :count] = sorted_zs[start:end]
        dzcats_out[pix, :count] = sorted_ddzs[start:end]
        dwcats_out[pix, :count] = sorted_wts[start:end]
        for ds in marks_out:
            marks_out[ds][pix, :count] = sorted_marks[ds][start:end]

    _end()

    # 6. Save outputs
    out_file = save_dir / f'catalog_pixelated_nside_{nside}.h5'
    _section("Saving outputs")
    with h5py.File(out_file, 'w') as f:
        f.attrs['nside'] = nside
        if opts.z_depth is not None:
            f.attrs['z_depth'] = float(opts.z_depth)
        f.create_dataset('zgals', data=cats_out, compression='gzip', shuffle=True)
        f.create_dataset('dzgals', data=dzcats_out, compression='gzip', shuffle=True)
        f.create_dataset('wgals', data=dwcats_out, compression='gzip', shuffle=True)
        f.create_dataset('ngals', data=counts, compression='gzip', shuffle=True)
        for ds, arr in marks_out.items():
            f.create_dataset(ds, data=arr, compression='gzip', shuffle=True)
    _ok(f"catalog  →  {out_file}")
    if marks_out:
        _ok(f"mark datasets: {', '.join(sorted(marks_out))}")
    if opts.z_depth is not None:
        _ok(f"survey z_depth attribute: {opts.z_depth}")
    _end()

    # 7. Plotting (Optional)
    if opts.add_plots:
        plot_diagnostics(counts, zs, npix, save_dir, nside)

    print()
    _banner("DONE")
    print()

if __name__ == "__main__":
    main()