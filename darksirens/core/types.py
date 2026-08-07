"""Core typed containers shared across darksirens packages."""
from typing import NamedTuple, Any


class CosmoParams(NamedTuple):
    """Cosmological parameters for the background universe."""
    H0: Any
    Om0: Any
    w0: Any = -1.0
    wa: Any = 0.0


class SurveyParams(NamedTuple):
    """Parameters dictating galaxy survey completeness and selection.

    ``n0`` is the comoving galaxy number density [Mpc^-3] of the n(z) =
    n0 (1+z)^delta model; with the unit-mass completeness kernels it is
    directly comparable to physical catalog densities.  ``z50`` and ``w``
    are the mock generator's roll-off truth and are never sampled (they carry
    no prior): the dark-siren completeness is the data-driven kernel ratio and
    does not use them.  ``b_miss`` is the LSS bias of the missing-galaxy density;
    ``alpha_miss`` enters only through the exact product
    ``alpha_miss * b_miss`` (perfect degeneracy) and defaults to 1 so that
    ``b_miss`` alone carries the modulation.  ``sigma_kde`` broadens the
    catalog redshift kernels in quadrature (an effective floor of 1e-4 in
    redshift, ~30 km/s, protects spectroscopic entries numerically).

    ``complete_empty_pixel_policy`` controls how the formally complete-catalog
    prior treats catalog rows with zero real galaxies: ``0`` is the strict
    policy (zero probability), while ``1`` enables a volume-prior robustness
    approximation.

    ``lss_corr_length_mpc``, ``lss_sigma`` and ``lss_corr_length_ang`` are
    **fixed** hyperparameters of the offline LSS-conditioned lognormal
    completion builder (:mod:`darksirens.redshift.lognormal_completion`): the radial
    comoving correlation length [Mpc], the latent-Gaussian field standard
    deviation, and (for the 3-D angular-coupling ``mode="gp3d"`` builder only)
    the **angular** correlation length in chordal units on the unit sphere
    (``ls_sph`` of the (sphere x z) GP; the chordal distance between two unit
    vectors lies in ``[0, 2]``, so an angular correlation length ``ell``
    corresponds to ``~2 arcsin(ell/2)``).  They parameterise the completion
    field and are **never marginalised over** — the GW likelihood does not read
    them; only the offline builder does (the bias of the field reuses
    ``b_miss``).  They are deliberately excluded from the sampled parameter
    space.

    ``wl_params`` defaults to ``None``, preserving existing behaviour for all
    non-weak-lensing universe models.  When non-``None`` it must be a
    ``darksirens.lensing.wlmagnification.WLParams`` instance carrying the
    weak-lensing magnification model used by ``universe_model='spectral_sirens_wl'``.

    ``z_depth`` is an optional per-survey redshift depth (a concrete Python
    float or ``None``, never a sampled/traced value) encoding the survey's
    completeness prior: beyond ``z_depth`` completeness is exactly zero, so every
    modeled host there is uncatalogued (missing, not nonexistent) -- the source
    prior above the depth relaxes to the volumetric x population shape and the
    observed catalog kernel is zeroed there (:mod:`darksirens.redshift.completion`,
    :mod:`darksirens.redshift.catalog`).  ``None`` means completeness is estimated
    over the full grid (the legacy, bit-identical path).

    ``c_mode`` selects the completeness estimator, encoded by the int enum
    :data:`darksirens.core.constants.C_MODES` and decoded eagerly pre-jit like
    ``complete_empty_pixel_policy``: ``per_pixel`` (0, the legacy DEFAULT,
    bit-identical -- the per-pixel matched-kernel ratio) or ``aggregate`` (1,
    ONE sky-aggregate curve ``Cbar(z)`` per proposal, broadcast to every
    pixel, so the observed angular clustering cannot be absorbed into the
    missing budget; see :mod:`darksirens.redshift.completion`).  Storage
    convention (the ``z_depth`` pattern): per_pixel is carried as ``None``
    (the default; a concrete ``0`` is also accepted and normalised eagerly),
    aggregate as the int ``1`` -- the branch is then STRUCTURAL (``None``-ness
    of a pytree leaf), so it survives nested jit boundaries where an int leaf
    becomes a value-unreadable tracer (``darksiren_log_likelihood`` takes the
    survey as a jit argument).  Never sampled/traced by intent.  A survey's
    ``c_mode`` must match the ``c_mode`` any prebuilt Q table was fit under
    (hard-checked at load; the two targets differ by the entire clustering
    signal).
    """
    n0: Any
    z50: Any
    w: Any
    delta: Any
    b_miss: Any
    alpha_miss: Any
    sigma_kde: Any = 0.0
    complete_empty_pixel_policy: Any = 0
    lss_corr_length_mpc: Any = 50.0
    lss_sigma: Any = 1.0
    lss_corr_length_ang: Any = 0.2
    wl_params: Any = None
    z_depth: Any = None
    c_mode: Any = None         # None/0=per_pixel (legacy), 1=aggregate; structural, never traced


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
    lss_completion_logq : (N_catalog_rows, N_grid) or (N_global_pix, N_grid) or None
        Deterministic log Q_LSS(p, z) completion table (preferred over the
        linear form).  Multiplies the missing-galaxy branch:
        ``dN_miss = (1 - C) dN_exp Q_LSS``, replacing the legacy
        ``max(1 + b_eff delta_g, 0)`` factor when supplied.  Built offline by
        :mod:`darksirens.redshift.lognormal_completion`; consumed deterministically
        (the likelihood never samples it).  ``N_grid`` must equal ``zgrid.size``.
    lss_completion_q : same shape as ``lss_completion_logq`` or None
        Linear-space Q_LSS table.  ``lss_completion_logq`` is preferred if both
        are supplied.
    lss_completion_logq_members : (M, N_rows, N_grid) or (M, N_global_pix, N_grid) or None
        Fixed posterior ensemble of log Q_LSS (M members).  Used only for
        Bayesian redshift-prior **diagnostics** (posterior-mean and member
        bands); not threaded through the per-event GW likelihood.
    lss_completion_q_members : same shape as ``lss_completion_logq_members`` or None
        Linear-space ensemble (``logq`` preferred if both supplied).
    lss_completion_indexing : int
        Indexing convention for the Q tables, stored as an int enum so the
        container stays a valid JAX pytree (a string leaf would break tracing):
        ``0`` = auto (infer from shape: first axis == N_catalog_rows ⇒ compact,
        else global), ``1`` = compact (index by catalog row), ``2`` = global
        (index by ``unique_pixels[row]``).  Members infer from the *second*
        axis when auto.  The human-readable string lives only in the HDF5 file.
    mark_logmstar, mark_logssfr, mark_metallicity, mark_color : (N_rows, N_max_gals) or None
        Optional per-galaxy "marks" — galaxy properties used by the marked-host
        model (:mod:`darksirens.marks`) to reweight the catalog's contribution to
        the redshift prior via a BBH-host efficiency ``h(marks | eta)``.  Stored
        **z-centered** (``x - E[x|z]``, computed on the full catalog at load) so
        the sampled ``eta`` measure host preference at fixed redshift.  Each field
        is ``None`` unless that mark was provided; ``mark_model="none"`` (default)
        ignores them entirely (bit-for-bit legacy behaviour).
    """
    apix: Any
    zgals: Any
    dzgals: Any
    wgals: Any
    ngals: Any
    delta_g_pix_z: Any
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
    # --- LSS-conditioned lognormal completion (optional; default None = legacy) ---
    lss_completion_logq: Any = None          # (N_rows|N_pix, N_grid) | None
    lss_completion_q: Any = None             # (N_rows|N_pix, N_grid) | None
    lss_completion_logq_members: Any = None  # (M, N_rows|N_pix, N_grid) | None
    lss_completion_q_members: Any = None     # (M, N_rows|N_pix, N_grid) | None
    lss_completion_indexing: Any = 0         # int enum: 0=auto, 1=compact, 2=global
    # --- marked-host model marks (optional, z-centered; default None = unused) ---
    mark_logmstar: Any = None       # (N_rows, N_max_gals) | None
    mark_logssfr: Any = None        # (N_rows, N_max_gals) | None
    mark_metallicity: Any = None    # (N_rows, N_max_gals) | None
    mark_color: Any = None          # (N_rows, N_max_gals) | None
    # --- FIELD-convention sky-weighting normalization inputs (optional) --------
    # Populated ONLY for ``--catalog_sky_weighting field`` (host-fraction
    # estimand); ``None`` (default) leaves the conditional per-pixel normalizer
    # path bit-identical.  They carry the survey-global ingredients needed to
    # build the GLOBAL normalizer Z_k(theta) = Sum_all-pixels [N_obs + N_miss],
    # replacing the per-pixel Z[pix] in the mixture weight so ``fcat_k`` measures
    # the host FRACTION (number-density / sky-clustering contrast) rather than
    # the per-pixel z-shape preference.  See
    # :func:`darksirens.redshift.completion.build_field_normalization_inputs` and
    # :func:`darksirens.redshift.completion.field_global_log_Z`.
    field_dN_obs_s: Any = None       # (n_occupied, N_grid) float32 smoothed observed density
    field_n_empty: Any = None        # scalar: number of empty (galaxy-free) survey pixels
    field_N_obs_total: Any = None    # scalar: total observed real-galaxy count over ALL pixels
    # --- FIELD-normalizer budget modulations (optional; default None = lss=1) --
    # Row-aligned with ``field_dN_obs_s`` via ``field_occupied_pixels`` so the
    # global normalizer carries the SAME per-pixel missing-budget factor as the
    # numerator: lss_p = Q_p(z) (deterministic Q_LSS) or
    # max(1 + b_eff*delta_g_p(z), 0) (local overdensity; empty pixels carry
    # delta_g == 0 by construction, so only occupied rows are stored).  Built by
    # :func:`darksirens.redshift.completion.build_field_lss_q_inputs` /
    # :func:`darksirens.redshift.completion.build_field_delta_g_inputs`.
    field_occupied_pixels: Any = None   # (n_occupied,) int32 global pixel ids
    field_lss_q: Any = None             # (n_occupied, N_grid) float32 LINEAR Q rows
    field_lss_q_empty_sum: Any = None   # (N_grid,) float64 Sum_empty Q_p(z) (data constant)
    field_delta_g: Any = None           # (n_occupied, N_grid) float32 delta_g rows
    # Per-member field normalizer inputs (Q ensemble marginalization under the
    # field convention): one modulation-row block + empty-pixel budget curve
    # per ensemble member, built by build_field_lss_q_member_inputs.
    field_lss_q_members: Any = None            # (M, n_occupied, N_grid) float32
    field_lss_q_empty_sum_members: Any = None  # (M, N_grid) float64
    # Marked-host field normalizer: flat FULL-SKY per-galaxy inputs so mu_miss
    # and the observed marked mass Sum w_i h_i are view-independent (PE and
    # selection share ONE global Z).  Built by build_field_mark_inputs.
    field_mark_z: Any = None            # (N_gal_total,) float32 galaxy redshifts
    field_mark_w: Any = None            # (N_gal_total,) float32 base weights
    field_mark_values: Any = None       # (N_gal_total, n_marks) float32 z-centred marks
    # DEPTH-truncated field normalizer: flat FULL-SKY per-galaxy kernel geometry
    # so the survey-global OBSERVED term carries the same depth convention as the
    # per-pixel numerator -- Sum_pix N_obs,pix * m_pix(theta), not the raw
    # field_N_obs_total (which counts above-depth catalogued galaxies twice: once
    # observed, again in the missing budget).  Populated ONLY when a
    # ``survey.z_depth`` is active; built by
    # :func:`darksirens.redshift.completion.build_field_depth_inputs`, consumed by
    # :func:`darksirens.redshift.completion.field_observed_global_total`.  Same
    # row-major galaxy order as the field_mark_* arrays.
    field_depth_z: Any = None           # (N_gal_total,) float64 galaxy redshifts
    field_depth_dz: Any = None          # (N_gal_total,) float64 per-galaxy sigma
    field_depth_c: Any = None           # (N_gal_total,) float64 N_obs,pix*w_i/W_pix


class GWEvent(NamedTuple):
    """
    JAX-compatible PyTree container for Gravitational Wave Parameter Estimation (PE) samples.
    Supports either a single event or a stacked batch of multiple events.

    Notes
    -----
    Do NOT construct directly — use ``darksirens.likelihood.events.make_gw_event``,
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
    # Sky direction unit vector n̂ = (cos δ cos α, cos δ sin α, sin δ), per sample.
    # Carried alongside dL so the angular (sky) model can be evaluated at full
    # resolution even with no galaxy catalog (where ``pixels`` is coarse).
    # Default None for back-compat with isotropic runs / direct construction;
    # ``make_gw_event`` always populates them (zeros when unspecified).
    nx: Any = None
    ny: Any = None
    nz: Any = None

    @property
    def chirp_mass(self):
        """Detector-frame chirp mass."""
        return (self.m1det * self.m2det) ** (3 / 5) / (self.m1det + self.m2det) ** (1 / 5)
