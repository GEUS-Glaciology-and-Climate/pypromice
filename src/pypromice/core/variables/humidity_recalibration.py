"""
rh_recal.envelope

Fixed-P95 temperature-dependent RH upper-envelope fitting.

Final method
------------
- Use only temperatures below 0 deg C.
- Use 2 deg C temperature bins.
- Require at least 40 observations for a statistical bin.
- Calculate P95 only.
- No adaptive P90/P95/P99 switching.
- No density-branch selection.
- If real observations extend colder than the first valid P95 bin,
  optionally use a sparse cold-tail support point.
- The cold-tail support point is NOT labelled P95.
- No artificial cold-tail blending toward 100 % saturation.
- Fit the supported envelope using PCHIP.
- Derive a bounded multiplicative RH correction factor.

Correction
----------
    factor(T) = physical_saturation / fitted_envelope(T)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


__all__ = [
    "EnvelopeConfig",
    "EnvelopeResult",
    "fit_envelope",
    "apply_correction",
]


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class EnvelopeConfig:
    """
    Configuration for fixed-P95 envelope fitting.
    """

    # Temperature range
    t_min: float = -70.0
    t_max: float = 0.0

    # Final scientific settings
    bin_width: float = 2.0
    percentile: float = 0.95
    min_points_per_bin: int = 40

    # Basic RH screening
    rh_min: float = 20.0
    rh_max: float = 130.0

    # Saturation target
    physical_saturation: float = 100.0

    # Maximum RH used only for the QC/display output.
    # The scientific rh_corrected variable remains uncapped.
    max_corrected_rh: float = 100.0

    # Correction-factor limits
    min_factor: float = 0.80
    max_factor: float = 1.35

    @property
    def min_correction_factor(self) -> float:
        """Backward-compatible alias for min_factor."""
        return self.min_factor

    @property
    def max_correction_factor(self) -> float:
        """Backward-compatible alias for max_factor."""
        return self.max_factor

    # Interpolation grid
    grid_points: int = 600

    # Sparse cold-tail support
    cold_tail_min_points: int = 8
    cold_tail_max_points: int = 30
    cold_tail_quantile: float = 0.50

    # --------------------------------------------------------
    # Compatibility with older config files
    # --------------------------------------------------------
    #
    # These are retained because default_config.toml may still
    # contain them. They are NOT used for adaptive percentile
    # selection.

    q_dense: float = 0.95
    q_sparse: float = 0.95
    sparse_threshold: int = 40

    @classmethod
    def from_dict(
        cls,
        values: dict | None,
    ) -> "EnvelopeConfig":
        """
        Construct EnvelopeConfig from a dictionary.

        Unknown old configuration options are ignored.

        The final method is forced to:
            bin width = 2 deg C
            percentile = P95
            minimum observations = 40
        """

        if not values:
            cfg = cls()

        else:
            valid_fields = set(
                cls.__dataclass_fields__.keys()
            )

            clean = {
                key: value
                for key, value in values.items()
                if key in valid_fields
            }

            cfg = cls(
                **clean
            )

        # ----------------------------------------------------
        # Force final scientific method
        # ----------------------------------------------------

        cfg.bin_width = 2.0
        cfg.percentile = 0.95
        cfg.min_points_per_bin = 40

        # Old fields are retained only for compatibility.
        cfg.q_dense = 0.95
        cfg.q_sparse = 0.95
        cfg.sparse_threshold = 40

        return cfg


# ============================================================
# OUTPUT CONTAINER
# ============================================================


@dataclass
class EnvelopeResult:
    """
    Output from fit_envelope().

    factor_at() provides the temperature-dependent correction
    factor required by correction.py.
    """

    grid_T: np.ndarray
    envelope: np.ndarray
    factor: np.ndarray
    binned: pd.DataFrame

    diagnostics: dict = field(
        default_factory=dict
    )

    def factor_at(
        self,
        temperature,
    ):
        """
        Evaluate the fitted correction factor at temperature.
        """

        values = np.asarray(
            temperature,
            dtype=float,
        )

        result = np.interp(
            values,
            self.grid_T,
            self.factor,
            left=float(self.factor[0]),
            right=1.0,
        )

        if np.ndim(temperature) == 0:
            return float(result)

        return result



# ============================================================
# APPLY RH CORRECTION
# ============================================================


def apply_correction(
    frame: pd.DataFrame,
    envelope: EnvelopeResult,
    cfg: EnvelopeConfig,
) -> pd.DataFrame:
    """
    Apply the fitted temperature-dependent multiplicative
    RH correction.

    The scientific corrected RH is kept uncapped.
    A separate capped variable is provided only for QC/display.
    """

    out = frame.copy()

    factor = envelope.factor_at(
        out["T"].to_numpy(dtype=float)
    )

    corrected = (
        out["rh"].to_numpy(dtype=float)
        * factor
    )

    out["rh_correction_factor"] = factor

    # Uncapped scientific result used for validation and RMSE.
    out["rh_corrected"] = corrected

    # Capped version used only for optional display or QC.
    out["rh_corrected_qc"] = np.clip(
        corrected,
        0.0,
        cfg.max_corrected_rh,
    )

    return out
# ============================================================
# PREPARE INPUT DATA
# ============================================================


def _prepare_data(
    frame: pd.DataFrame,
    cfg: EnvelopeConfig,
) -> pd.DataFrame:
    """
    Prepare valid subfreezing T/RH observations.
    """

    required = {
        "T",
        "rh",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            "Envelope input is missing columns: "
            f"{sorted(missing)}"
        )

    cold = frame[
        [
            "T",
            "rh",
        ]
    ].copy()

    # Convert to numeric
    cold["T"] = pd.to_numeric(
        cold["T"],
        errors="coerce",
    )

    cold["rh"] = pd.to_numeric(
        cold["rh"],
        errors="coerce",
    )

    # Remove invalid values
    cold = cold.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    cold = cold.dropna(
        subset=[
            "T",
            "rh",
        ]
    )

    # Subfreezing data only
    cold = cold[
        (cold["T"] >= cfg.t_min)
        &
        (cold["T"] < cfg.t_max)
    ]

    # Remove gross RH outliers
    cold = cold[
        cold["rh"].between(
            cfg.rh_min,
            cfg.rh_max,
        )
    ]

    cold = (
        cold
        .sort_values("T")
        .reset_index(drop=True)
    )

    if cold.empty:
        raise ValueError(
            "No valid subfreezing RH observations "
            "available for envelope fitting"
        )

    return cold


# ============================================================
# FIXED P95 TEMPERATURE BINS
# ============================================================


def _calculate_p95_bins(
    cold: pd.DataFrame,
    cfg: EnvelopeConfig,
) -> pd.DataFrame:
    """
    Calculate fixed P95 in 2 deg C temperature bins.

    Only bins containing at least 40 observations are accepted.
    """

    edges = np.arange(
        cfg.t_min,
        cfg.t_max
        + cfg.bin_width,
        cfg.bin_width,
    )

    working = cold.copy()

    working["T_bin"] = pd.cut(
        working["T"],
        bins=edges,
        include_lowest=True,
    )

    rows: list[dict] = []

    for interval, group in working.groupby(
        "T_bin",
        observed=True,
    ):

        n = len(group)

        # Minimum sample criterion
        if n < cfg.min_points_per_bin:
            continue

        rh_values = group[
            "rh"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # P95 ONLY
        # ----------------------------------------------------

        p95 = float(
            np.quantile(
                rh_values,
                0.95,
            )
        )

        rows.append(
            {
                "T_mid": float(
                    interval.mid
                ),

                "T_left": float(
                    interval.left
                ),

                "T_right": float(
                    interval.right
                ),

                "n": int(n),

                "q_used": 0.95,

                "p95": p95,

                "selected": p95,

                "point_type": "P95",
            }
        )

    if len(rows) < 2:
        raise ValueError(
            "Too few populated temperature bins "
            "for fixed-P95 envelope fitting"
        )

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values("T_mid")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# COLD-TAIL SUPPORT
# ============================================================


def _cold_tail_support(
    cold: pd.DataFrame,
    p95_bins: pd.DataFrame,
    cfg: EnvelopeConfig,
) -> dict | None:
    """
    Construct one sparse observational support point when
    observations extend colder than the first statistically
    accepted P95 bin.

    IMPORTANT
    ---------
    This point is not P95.

    It is a support point based on real observations in a
    temperature region where there are not enough observations
    to estimate P95 reliably.

    The point is NOT subsequently forced toward 100 %
    saturation.
    """

    if p95_bins.empty:
        return None

    # Cold boundary of first accepted P95 bin
    first_left = float(
        p95_bins.iloc[0][
            "T_left"
        ]
    )

    # Select real observations colder than the first P95 bin
    tail = cold[
        cold["T"] < first_left
    ].copy()

    if len(tail) < cfg.cold_tail_min_points:
        return None

    # Coldest observations first
    tail = tail.sort_values(
        "T"
    )

    # Restrict support estimation to the coldest observations
    tail = tail.head(
        cfg.cold_tail_max_points
    )

    if len(tail) < cfg.cold_tail_min_points:
        return None

    temperatures = tail[
        "T"
    ].to_numpy(
        dtype=float
    )

    rh_values = tail[
        "rh"
    ].to_numpy(
        dtype=float
    )

    # Representative cold-tail temperature
    support_temperature = float(
        np.median(
            temperatures
        )
    )

    # Representative observational RH support
    support_rh = float(
        np.quantile(
            rh_values,
            cfg.cold_tail_quantile,
        )
    )

    return {
        "T_mid": support_temperature,

        "T_left": float(
            np.min(temperatures)
        ),

        "T_right": float(
            np.max(temperatures)
        ),

        "n": int(
            len(tail)
        ),

        "q_used": np.nan,

        "p95": np.nan,

        "selected": support_rh,

        "point_type":
            "cold_tail_support",
    }


# ============================================================
# BUILD ENVELOPE SUPPORT
# ============================================================


def _build_support_table(
    cold: pd.DataFrame,
    p95_bins: pd.DataFrame,
    cfg: EnvelopeConfig,
) -> pd.DataFrame:
    """
    Build the support table containing:

    - fixed P95 bins
    - optional observational cold-tail support
    """

    support = p95_bins.copy()

    tail = _cold_tail_support(
        cold,
        p95_bins,
        cfg,
    )

    if tail is not None:

        tail_frame = pd.DataFrame(
            [tail]
        )

        support = pd.concat(
            [
                tail_frame,
                support,
            ],
            ignore_index=True,
        )

    support = (
        support
        .sort_values("T_mid")
        .drop_duplicates(
            subset=[
                "T_mid",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return support


# ============================================================
# PCHIP INTERPOLATION
# ============================================================


def _fit_pchip(
    support: pd.DataFrame,
    cfg: EnvelopeConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Fit a shape-preserving PCHIP curve through the support
    points.

    Extrapolation beyond supported temperatures is disabled.

    No post-PCHIP blending toward physical saturation is
    applied. The cold end therefore remains controlled by the
    observational cold-tail support point and the adjacent
    fixed-P95 bins.
    """

    x = support[
        "T_mid"
    ].to_numpy(
        dtype=float
    )

    y = support[
        "selected"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    if x.size < 2:
        raise ValueError(
            "At least two envelope support points "
            "are required for PCHIP fitting"
        )

    # Sort by temperature
    order = np.argsort(
        x
    )

    x = x[
        order
    ]

    y = y[
        order
    ]

    # Remove duplicate temperatures
    unique_x, indices = np.unique(
        x,
        return_index=True,
    )

    x = unique_x

    y = y[
        indices
    ]

    if x.size < 2:
        raise ValueError(
            "Envelope support temperatures "
            "are not sufficiently distinct"
        )

    # Shape-preserving PCHIP
    interpolator = PchipInterpolator(
        x,
        y,
        extrapolate=False,
    )

    t_min_supported = float(
        np.min(x)
    )

    t_max_supported = float(
        np.max(x)
    )

    n_grid = max(
        int(cfg.grid_points),
        100,
    )

    grid_T = np.linspace(
        t_min_supported,
        t_max_supported,
        n_grid,
    )

    envelope = interpolator(
        grid_T
    )

    envelope = np.asarray(
        envelope,
        dtype=float,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do not blend the cold end toward 100 %.
    #
    # The cold end is already supported by real observations
    # through _cold_tail_support(). PCHIP connects that support
    # smoothly to the first statistically valid P95 bin.
    # --------------------------------------------------------

    return (
        grid_T,
        envelope,
    )


# ============================================================
# CORRECTION FACTOR
# ============================================================


def _calculate_factor(
    envelope: np.ndarray,
    cfg: EnvelopeConfig,
) -> np.ndarray:
    """
    Calculate the RH correction factor:

        factor = physical_saturation / envelope
    """

    envelope = np.asarray(
        envelope,
        dtype=float,
    )

    safe = envelope.copy()

    # Remove invalid envelope values
    safe[
        ~np.isfinite(safe)
    ] = np.nan

    safe[
        safe <= 0
    ] = np.nan

    factor = (
        cfg.physical_saturation
        /
        safe
    )

    # Keep correction bounded
    factor = np.clip(
        factor,
        cfg.min_factor,
        cfg.max_factor,
    )

    return factor


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================


def fit_envelope(
    frame: pd.DataFrame,
    cfg: EnvelopeConfig,
) -> EnvelopeResult:
    """
    Fit the fixed-P95 RH upper envelope.

    Parameters
    ----------
    frame
        DataFrame containing at least:

            T
            rh

    cfg
        EnvelopeConfig instance.

    Returns
    -------
    EnvelopeResult
        grid_T
        envelope
        factor
        binned
        diagnostics
    """

    # ========================================================
    # FORCE THE FINAL METHOD
    # ========================================================

    cfg.bin_width = 2.0
    cfg.percentile = 0.95
    cfg.min_points_per_bin = 40

    cfg.q_dense = 0.95
    cfg.q_sparse = 0.95
    cfg.sparse_threshold = 40

    # ========================================================
    # PREPARE VALID SUBFREEZING OBSERVATIONS
    # ========================================================

    cold = _prepare_data(
        frame,
        cfg,
    )

    # ========================================================
    # CALCULATE TRUE P95 BINS
    # ========================================================

    p95_bins = _calculate_p95_bins(
        cold,
        cfg,
    )

    # ========================================================
    # OPTIONAL OBSERVATIONAL COLD-TAIL SUPPORT
    # ========================================================

    support = _build_support_table(
        cold,
        p95_bins,
        cfg,
    )

    # ========================================================
    # FIT PCHIP ENVELOPE
    # ========================================================

    grid_T, envelope = _fit_pchip(
        support,
        cfg,
    )

    # ========================================================
    # CALCULATE CORRECTION FACTOR
    # ========================================================

    factor = _calculate_factor(
        envelope,
        cfg,
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    p95_rows = support[
        support["point_type"]
        ==
        "P95"
    ]

    tail_rows = support[
        support["point_type"]
        ==
        "cold_tail_support"
    ]

    diagnostics = {
        "method":
            "fixed_P95",

        "percentile":
            0.95,

        "bin_width_degC":
            2.0,

        "min_points_per_bin":
            40,

        "n_input_subfreezing":
            int(
                len(cold)
            ),

        "n_p95_bins":
            int(
                len(p95_rows)
            ),

        "cold_tail_used":
            bool(
                not tail_rows.empty
            ),

        "observed_temperature_min":
            float(
                cold["T"].min()
            ),

        "observed_temperature_max":
            float(
                cold["T"].max()
            ),

        "support_temperature_min":
            float(
                support["T_mid"].min()
            ),

        "support_temperature_max":
            float(
                support["T_mid"].max()
            ),

        "factor_min":
            float(
                np.nanmin(factor)
            ),

        "factor_max":
            float(
                np.nanmax(factor)
            ),
    }

    # --------------------------------------------------------
    # Cold-tail diagnostics when used
    # --------------------------------------------------------

    if not tail_rows.empty:

        tail_row = tail_rows.iloc[
            0
        ]

        diagnostics[
            "cold_tail_temperature"
        ] = float(
            tail_row[
                "T_mid"
            ]
        )

        diagnostics[
            "cold_tail_rh"
        ] = float(
            tail_row[
                "selected"
            ]
        )

        diagnostics[
            "cold_tail_n"
        ] = int(
            tail_row[
                "n"
            ]
        )

    # ========================================================
    # RETURN
    # ========================================================

    return EnvelopeResult(
        grid_T=grid_T,
        envelope=envelope,
        factor=factor,
        binned=support,
        diagnostics=diagnostics,
    )