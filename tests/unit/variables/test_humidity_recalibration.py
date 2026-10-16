import numpy as np
import pandas as pd
import pytest

from pypromice.core.variables.humidity_recalibration import (
    EnvelopeConfig,
    EnvelopeResult,
    apply_correction,
    fit_envelope,
)


def make_two_bin_frame():
    """
    Deterministic input containing two statistically valid
    2 deg C temperature bins with 40 observations each.
    """

    t_bin_1 = np.full(40, -5.0)
    t_bin_2 = np.full(40, -3.0)

    rh_bin_1 = np.linspace(80.0, 100.0, 40)
    rh_bin_2 = np.linspace(85.0, 105.0, 40)

    return pd.DataFrame(
        {
            "T": np.concatenate(
                [
                    t_bin_1,
                    t_bin_2,
                ]
            ),
            "rh": np.concatenate(
                [
                    rh_bin_1,
                    rh_bin_2,
                ]
            ),
        }
    )


def test_fixed_p95_bins():
    """
    Valid bins must use fixed P95 with at least 40 observations.
    """

    frame = make_two_bin_frame()
    cfg = EnvelopeConfig()

    result = fit_envelope(
        frame,
        cfg,
    )

    p95_rows = result.binned[
        result.binned["point_type"] == "P95"
    ]

    assert len(p95_rows) == 2

    assert np.all(
        p95_rows["n"].to_numpy()
        == 40
    )

    assert np.allclose(
        p95_rows["q_used"].to_numpy(dtype=float),
        0.95,
    )

    assert np.allclose(
        p95_rows["selected"].to_numpy(dtype=float),
        p95_rows["p95"].to_numpy(dtype=float),
    )


def test_bins_below_minimum_count_are_rejected():
    """
    Fewer than 40 observations per bin must not produce
    a statistically valid P95 envelope.
    """

    frame = pd.DataFrame(
        {
            "T": np.concatenate(
                [
                    np.full(39, -5.0),
                    np.full(39, -3.0),
                ]
            ),
            "rh": np.concatenate(
                [
                    np.linspace(80.0, 100.0, 39),
                    np.linspace(85.0, 105.0, 39),
                ]
            ),
        }
    )

    cfg = EnvelopeConfig()

    with pytest.raises(
        ValueError,
        match="Too few populated temperature bins",
    ):
        fit_envelope(
            frame,
            cfg,
        )


def test_cold_tail_support_is_observational_not_p95():
    """
    Sparse cold observations should create an observational
    support point and must not be labelled P95.
    """

    base = make_two_bin_frame()

    cold_tail = pd.DataFrame(
        {
            "T": np.linspace(
                -20.0,
                -10.0,
                10,
            ),
            "rh": np.linspace(
                70.0,
                79.0,
                10,
            ),
        }
    )

    frame = pd.concat(
        [
            cold_tail,
            base,
        ],
        ignore_index=True,
    )

    cfg = EnvelopeConfig()

    result = fit_envelope(
        frame,
        cfg,
    )

    tail = result.binned[
        result.binned["point_type"]
        == "cold_tail_support"
    ]

    assert len(tail) == 1

    row = tail.iloc[0]

    assert np.isnan(
        float(row["q_used"])
    )

    assert np.isnan(
        float(row["p95"])
    )

    assert np.isfinite(
        float(row["selected"])
    )

    assert result.diagnostics[
        "cold_tail_used"
    ] is True


def test_no_cold_tail_when_too_few_observations():
    """
    Fewer than cold_tail_min_points observations must not
    create a cold-tail support point.
    """

    base = make_two_bin_frame()

    cold_tail = pd.DataFrame(
        {
            "T": np.linspace(
                -20.0,
                -10.0,
                7,
            ),
            "rh": np.linspace(
                70.0,
                76.0,
                7,
            ),
        }
    )

    frame = pd.concat(
        [
            cold_tail,
            base,
        ],
        ignore_index=True,
    )

    cfg = EnvelopeConfig()

    result = fit_envelope(
        frame,
        cfg,
    )

    assert (
        result.binned["point_type"]
        == "cold_tail_support"
    ).sum() == 0

    assert result.diagnostics[
        "cold_tail_used"
    ] is False


def test_factor_matches_physical_saturation_over_envelope():
    """
    The correction factor must equal 100 / fitted envelope
    wherever the raw value lies inside the configured bounds.
    """

    frame = make_two_bin_frame()
    cfg = EnvelopeConfig()

    result = fit_envelope(
        frame,
        cfg,
    )

    expected = (
        cfg.physical_saturation
        / result.envelope
    )

    expected = np.clip(
        expected,
        cfg.min_factor,
        cfg.max_factor,
    )

    assert np.allclose(
        result.factor,
        expected,
    )


def test_factor_stays_within_configured_bounds():
    """
    Every fitted correction factor must respect the configured
    minimum and maximum factor limits.
    """

    frame = make_two_bin_frame()
    cfg = EnvelopeConfig()

    result = fit_envelope(
        frame,
        cfg,
    )

    assert np.nanmin(
        result.factor
    ) >= cfg.min_factor

    assert np.nanmax(
        result.factor
    ) <= cfg.max_factor


def test_factor_at_temperature_boundaries():
    """
    Colder temperatures use the coldest fitted factor.
    Warmer temperatures receive no correction.
    """

    envelope = EnvelopeResult(
        grid_T=np.array(
            [
                -20.0,
                -10.0,
            ]
        ),
        envelope=np.array(
            [
                80.0,
                100.0,
            ]
        ),
        factor=np.array(
            [
                1.25,
                1.0,
            ]
        ),
        binned=pd.DataFrame(),
    )

    values = envelope.factor_at(
        np.array(
            [
                -30.0,
                -15.0,
                5.0,
            ]
        )
    )

    assert values[0] == pytest.approx(
        1.25
    )

    assert values[1] == pytest.approx(
        1.125
    )

    assert values[2] == pytest.approx(
        1.0
    )


def test_apply_correction_keeps_scientific_result_uncapped():
    """
    Scientific RH remains uncapped while the separate QC
    variable is capped at max_corrected_rh.
    """

    envelope = EnvelopeResult(
        grid_T=np.array(
            [
                -20.0,
                -10.0,
            ]
        ),
        envelope=np.array(
            [
                80.0,
                80.0,
            ]
        ),
        factor=np.array(
            [
                1.25,
                1.25,
            ]
        ),
        binned=pd.DataFrame(),
    )

    frame = pd.DataFrame(
        {
            "T": [
                -15.0,
            ],
            "rh": [
                90.0,
            ],
        }
    )

    cfg = EnvelopeConfig()

    corrected = apply_correction(
        frame,
        envelope,
        cfg,
    )

    assert corrected.loc[
        0,
        "rh_correction_factor",
    ] == pytest.approx(
        1.25
    )

    assert corrected.loc[
        0,
        "rh_corrected",
    ] == pytest.approx(
        112.5
    )

    assert corrected.loc[
        0,
        "rh_corrected_qc",
    ] == pytest.approx(
        100.0
    )


def test_invalid_input_columns_raise_key_error():
    """
    Envelope input must contain both T and rh.
    """

    frame = pd.DataFrame(
        {
            "temperature": [
                -10.0,
            ],
            "humidity": [
                90.0,
            ],
        }
    )

    cfg = EnvelopeConfig()

    with pytest.raises(
        KeyError,
        match="Envelope input is missing columns",
    ):
        fit_envelope(
            frame,
            cfg,
        )


def test_no_valid_subfreezing_data_raises_value_error():
    """
    A dataset containing no valid subfreezing observations
    cannot be used for envelope fitting.
    """

    frame = pd.DataFrame(
        {
            "T": [
                1.0,
                5.0,
                10.0,
            ],
            "rh": [
                80.0,
                90.0,
                100.0,
            ],
        }
    )

    cfg = EnvelopeConfig()

    with pytest.raises(
        ValueError,
        match="No valid subfreezing RH observations",
    ):
        fit_envelope(
            frame,
            cfg,
        )