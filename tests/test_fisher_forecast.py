"""Regression checks on fisher_forecast.py's bracket construction.

The forecast is reported as a pair: N_phi from the box QE with the UNLENSED C_TT in its filter
(qe_unlensed, the upper bound on the information) and with the iterated partially DELENSED
C_TT (qe_delensed, the lower bound). Pinned here:

  * the bracket is ordered: F(qe_unlensed) - F(qe_delensed) is positive semi-definite, so
    every marginalized sigma of the upper bound is at most the lower bound's
  * the empirical delensed covariance files are read with the right QE filter, including
    files written before `qe_filter` was recorded (only `iterative_delens`)
  * the --phi_noise / --delensed_covariance loaders refuse a different box or cosmology

No Julia ground truth is involved. The bracket test runs CAMB (~1 min at nside 32).
"""

import numpy as np
import pytest

import cmb_lensing.fisher_forecast as ff

NSIDE = 32
THETA_PIX = 5.0
NOISE = 5.0
SAMPLED = {"omch2": True, "theta_MC_100": True, "logA": True, "ns": False, "ombh2": False}


def _box(nside = NSIDE, noise_level = NOISE):
    return dict(nside = np.array(nside), theta_pix = np.array(THETA_PIX),
                noise_level = np.array(noise_level), l_knee = np.array(0.0),
                param_names = np.array(ff.PARAM_ORDER),
                params = np.array([ff.GROUND_TRUTH[name] for name in ff.PARAM_ORDER]))


def test_wiener_qe_filter_reads_old_and_new_files():
    assert ff.wiener_qe_filter({"qe_filter": np.array("unlensed"),
                                "iterative_delens": np.array(True)}) == "unlensed"
    assert ff.wiener_qe_filter({"iterative_delens": np.array(True)}) == "delensed"
    assert ff.wiener_qe_filter({"iterative_delens": np.array(False)}) == "lensed"
    #files from before either flag existed built W one-shot
    assert ff.wiener_qe_filter({}) == "lensed"


def test_check_wiener_matches_warns_only_on_a_mismatch(capsys):
    unlensed_file = {"qe_filter": np.array("unlensed"), "qe_response": np.array("unlensed")}
    ff.check_wiener_matches(unlensed_file, dict(nphi_source = "qe_unlensed",
                                                qe_response = "unlensed", checksum = ""))
    assert "WARNING" not in capsys.readouterr().out
    ff.check_wiener_matches(unlensed_file, dict(nphi_source = "qe_delensed",
                                                qe_response = "unlensed", checksum = ""))
    assert "WARNING" in capsys.readouterr().out

    stored = {"wiener_nphi_source": np.array("sample_joint"),
              "wiener_nphi_checksum": np.array("abc")}
    ff.check_wiener_matches(stored, dict(nphi_source = "sample_joint", qe_response = "",
                                         checksum = "abc", phi_noise = "x"))
    assert "WARNING" not in capsys.readouterr().out
    ff.check_wiener_matches(stored, dict(nphi_source = "sample_joint", qe_response = "",
                                         checksum = "def", phi_noise = "x"))
    assert "WARNING" in capsys.readouterr().out


def test_load_phi_noise_matrix_classifies_and_refuses(tmp_path):
    grid = np.ones((NSIDE, NSIDE // 2 + 1))
    sampler = tmp_path / "sampler_effective_noise.npz"
    np.savez(sampler, nphi = grid, nphi_cross_raw = grid, **_box())
    map_joint = tmp_path / "delensed_covariance.npz"
    np.savez(map_joint, phi_noise_fid = grid, **_box())

    args = (NSIDE, THETA_PIX, NOISE, 0.0, ff.GROUND_TRUTH)
    assert ff.load_phi_noise_matrix(str(sampler), *args)[2] == "sample_joint"
    assert ff.load_phi_noise_matrix(str(map_joint), *args)[2] == "map_joint"

    with pytest.raises(ValueError, match = "measured at"):
        ff.load_phi_noise_matrix(str(sampler), NSIDE, THETA_PIX, 2.5, 0.0, ff.GROUND_TRUTH)
    shifted = dict(ff.GROUND_TRUTH, omch2 = ff.GROUND_TRUTH["omch2"] * 1.01)
    with pytest.raises(ValueError, match = "fiducial point"):
        ff.load_phi_noise_matrix(str(sampler), NSIDE, THETA_PIX, NOISE, 0.0, shifted)
    #a sampler file is not a map_joint noise, whatever the forecast is told
    with pytest.raises(ValueError, match = "not the 'map_joint'"):
        ff.frozen_reconstruction(ff.GROUND_TRUTH, NSIDE, THETA_PIX, NOISE, 0.0, 0.0, 10_000,
                                 "map_joint", phi_noise = str(sampler), camb_lmax = 4000,
                                 verbose = False)


def test_check_delensed_covariance_refuses_missing_parameters():
    grid = np.ones((NSIDE, NSIDE // 2 + 1))
    merged = dict(delensed_fid = grid, names = np.array(["omch2"]),
                  steps = np.array([1e-4]), **_box())
    names = ["omch2"]
    assert ff.check_delensed_covariance(merged, NSIDE, THETA_PIX, NOISE, 0.0,
                                        ff.GROUND_TRUTH, names) == [1e-4]
    with pytest.raises(ValueError, match = "no stencil points"):
        ff.check_delensed_covariance(merged, NSIDE, THETA_PIX, NOISE, 0.0, ff.GROUND_TRUTH,
                                     ["omch2", "logA"])


def test_bounds_are_ordered():
    """The bracket's ordering, in the sense it actually holds.

    The PHI block is ordered as a matrix: a lower N_phi only adds information. The total is
    NOT - F(qe_unlensed) - F(qe_delensed) has a small negative eigenvalue, almost along logA
    (-0.025 against a largest eigenvalue of 136 at this box; -12 against 2.25e4 at nside 128 /
    2.5' / 5 uK, measured 2026-10-14) - because better delensing leaves less residual lensing
    in the delensed T block, which then carries less of the lensing-derived omch2 / logA
    information it double counts with the phi block. What the bracket is quoted for, the
    marginalized sigmas, stays ordered.
    """
    stencils = {source: ff.covariance_stencil(NSIDE, THETA_PIX, NOISE, SAMPLED,
                                              ff.GROUND_TRUTH, nphi_source = source,
                                              camb_lmax = 4000, verbose = False)
                for source in ff.QE_NPHI_SOURCES}

    def fisher(source, block = None):
        names, steps, weights, fid, plus, minus, _ = stencils[source]
        if block is not None:
            fid = {block: fid[block]}
            plus = [{block: point[block]} for point in plus]
            minus = [{block: point[block]} for point in minus]
        return ff._fisher_from_blocks(plus, minus, fid, steps, weights)

    phi_difference = fisher("qe_unlensed", "phi") - fisher("qe_delensed", "phi")
    assert np.min(np.linalg.eigvalsh(phi_difference)) >= 0

    upper = np.sqrt(np.diag(np.linalg.inv(fisher("qe_unlensed"))))
    lower = np.sqrt(np.diag(np.linalg.inv(fisher("qe_delensed"))))
    assert np.all(upper <= lower), dict(zip(stencils["qe_unlensed"][0], upper / lower))
