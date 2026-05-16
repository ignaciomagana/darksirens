# darksirens/utils/containers.py
from typing import NamedTuple, Any
import jax.numpy as jnp


class CosmoParams(NamedTuple):
    """Cosmological parameters for the background universe."""
    H0: Any
    Om0: Any


class SurveyParams(NamedTuple):
    """Parameters dictating galaxy survey completeness and selection.

    ``complete_empty_pixel_policy`` controls how the formally complete-catalog
    prior treats catalog rows with zero real galaxies: ``0`` is the strict
    policy (zero probability), while ``1`` enables a volume-prior robustness
    approximation.
    """
    n0: Any
    z50: Any
    w: Any
    delta: Any
    b_miss: Any
    alpha_miss: Any
    complete_empty_pixel_policy: Any = 0


class EMCatalog(NamedTuple):
    """
    EM galaxy catalog and precomputed grids for the redshift prior.

    Fields
    ------
    apix : float
        Solid angle per HEALPix pixel [sr].
    zgals : (N_catalog_rows, N_max_gals)
        Padded galaxy redshifts per catalog row.  For compact/sliced
        catalogs this has one row per unique inference pixel; for legacy
        full catalogs the row index is the global HEALPix pixel.
    dzgals : (N_catalog_rows, N_max_gals)
        Padded galaxy photo-z uncertainties.
    wgals : (N_catalog_rows, N_max_gals)
        Padded galaxy base weights (luminosity / completeness).
    ngals : (N_catalog_rows,)
        Number of real (non-padded) galaxies per catalog row.
    delta_g_pix_z : (N_pix, N_grid) or (1, N_grid)
        LSS overdensity field pre-computed on the redshift grid.  This
        remains globally pixel-indexed when LSS is enabled; compact
        catalogs use ``unique_pixels[row]`` before indexing it.
    sigma_kernel : float
        KDE bandwidth for the catalog prior [redshift units].
    dN_obs_kde : (N_unique_pix, N_grid) or None
        Precomputed per-pixel KDE grids for dN_obs/dz.
        None until ``build_pixel_kde_cache`` is called.
        If None, ``_catalog_completion_inner`` recomputes on the fly
        (correct but slower — only for backward compatibility).
    pixel_to_cache_idx : (N_pix_catalog,) or (N_catalog_rows,) or None
        Maps the incoming pixel/catalog-row index to a row in
        ``dN_obs_kde``.  Pixels not covered by the cache map to 0
        (never visited during inference, so the value is immaterial).
    unique_pixels : (N_catalog_rows,) or None
        Global HEALPix pixel represented by each compact catalog row.
        ``None`` means the catalog rows are already global pixel rows.
    sample_to_unique_idx : array or None
        Per-sample lookup map from the original PE/selection sample order to
        compact catalog row.  The likelihood passes this array as the
        GWEvent pixel index for compact catalogs; the field is retained on
        the catalog for diagnostics/introspection.
    counterpart_pixel : int or None
        Backward-compatible scalar global HEALPix pixel for a one-event
        bright-siren counterpart.
    counterpart_pixels : array or None
        Per-event global HEALPix pixels for bright-siren counterparts.
    counterpart_zs : array or None
        Per-event counterpart redshifts.
    counterpart_dzs : array or None
        Per-event Gaussian redshift uncertainties.
    active_counterpart_index : int
        Event index selected by the PE likelihood when evaluating a multi-event
        bright-siren prior.
    bright_siren_sky_marginalized : bool
        If true, the bright-siren prior applies only the counterpart redshift
        prior and does not require a GW sample to fall in the event counterpart
        pixel.
    """
    apix: Any
    zgals: Any
    dzgals: Any
    wgals: Any
    ngals: Any
    delta_g_pix_z: Any
    sigma_kernel: Any
    dN_obs_kde: Any            # (N_unique_pix, N_grid) | None
    pixel_to_cache_idx: Any    # (N_pix_catalog or N_catalog_rows,) | None
    unique_pixels: Any = None  # (N_catalog_rows,) | None
    sample_to_unique_idx: Any = None  # sample-shaped int array | None
    counterpart_pixel: Any = None  # legacy scalar global HEALPix pixel | None
    counterpart_pixels: Any = None  # per-event global HEALPix pixels | None
    counterpart_zs: Any = None  # per-event bright-siren redshifts | None
    counterpart_dzs: Any = None  # per-event redshift uncertainties | None
    active_counterpart_index: Any = 0  # event row selected by likelihood
    bright_siren_sky_marginalized: Any = False


class GWEvent(NamedTuple):
    """
    JAX-compatible PyTree container for Gravitational Wave Parameter Estimation (PE) samples.
    Supports either a single event or a stacked batch of multiple events.

    Notes
    -----
    Do NOT construct directly — use ``darksirens.inference.events.make_gw_event``,
    which applies ``lax.optimization_barrier`` to every field and pre-computes ``q``
    so it is never recomputed inside a vmap or ``lax.scan`` hot path.
    """
    m1det: Any      # Primary mass in the detector frame [M_sun]
    m2det: Any      # Secondary mass in the detector frame [M_sun]
    dL: Any         # Luminosity distance [Mpc]
    chieff: Any     # Effective inspiral spin parameter
    prior_wt: Any   # PE prior weights evaluated at the samples
    pixels: Any     # HEALPix pixel indices corresponding to the sky location
    q: Any          # Mass ratio m2det/m1det — stored at construction, never recomputed
    valid: Any      # Explicit structural mask; False for padded sentinel rows

    @property
    def chirp_mass(self):
        """Detector-frame chirp mass."""
        return (self.m1det * self.m2det) ** (3 / 5) / (self.m1det + self.m2det) ** (1 / 5)