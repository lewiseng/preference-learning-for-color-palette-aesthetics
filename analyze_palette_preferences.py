import os

import numpy as np
import pandas as pd

# Ensure Matplotlib can write its config/cache in restricted environments
_mplconfig_dir = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.environ.setdefault("MPLCONFIGDIR", _mplconfig_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv as mpl_rgb_to_hsv
from sklearn.model_selection import train_test_split  # noqa: F401 (imported for future use)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from scipy.special import i0
from typing import Dict, List, Optional, Tuple

# Default number of top-weighted Mturk features to keep
TOP_K_MTURK_FEATURES = 20


def loadData(matPath: str = "data/mturkData.mat") -> pd.DataFrame:
    """
    Load MTurk color palette ratings from a .mat file into a DataFrame.

    The resulting DataFrame contains all variables from the MAT file,
    with the following columns:
    - ids
    - names
    - targets (original mean ratings from the file)
    - userNormalizedTargets
    - color1_r, color1_g, color1_b, ..., color5_r, color5_g, color5_b
    """
    matData = loadmat(matPath)

    # Extract core arrays
    targets = matData.get("targets")
    ids = matData.get("ids")
    names = matData.get("names")
    data = matData.get("data")
    userNormalizedTargets = matData.get("userNormalizedTargets")

    if targets is None or ids is None or names is None or data is None:
        raise ValueError("Expected keys ('targets', 'ids', 'names', 'data') not found in MAT file.")

    nSamples = targets.shape[0]

    # Flatten simple column vectors
    targetsFlat = targets.reshape(nSamples)
    idsFlat = ids.reshape(nSamples)
    userNormFlat = userNormalizedTargets.reshape(nSamples) if userNormalizedTargets is not None else None

    # Convert names (object array of 1-element arrays) to plain strings
    nameList = []
    for row in names:
        # Each row is something like [array(['catamaran'], dtype='<U9')]
        cell = row[0]
        if isinstance(cell, np.ndarray):
            nameList.append(str(cell[0]))
        else:
            nameList.append(str(cell))

    # Flatten color data: shape (N, 5, 3) -> columns color1_r ... color5_b
    if data.ndim != 3 or data.shape[1] != 5 or data.shape[2] != 3:
        raise ValueError(f"Unexpected 'data' shape {data.shape}; expected (N, 5, 3).")

    colorCols = {}
    channelNames = ["r", "g", "b"]
    nColors = data.shape[1]
    for colorIdx in range(nColors):
        for chIdx, chName in enumerate(channelNames):
            colName = f"color{colorIdx + 1}_{chName}"
            colorCols[colName] = data[:, colorIdx, chIdx]

    # Build DataFrame with all information
    dfDict = {
        "ids": idsFlat,
        "names": nameList,
        "targets": targetsFlat,
    }

    if userNormFlat is not None:
        dfDict["userNormalizedTargets"] = userNormFlat

    dfDict.update(colorCols)

    df = pd.DataFrame(dfDict)
    return df


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to Lab, following the MATLAB RGB2Lab.m implementation.

    Args:
        rgb: Array of shape (3, num_colors) or (num_colors, 3).

    Returns:
        Lab array with shape (3, num_colors).
    """
    arr = np.asarray(rgb, dtype=float)
    if arr.ndim != 2 or 3 not in arr.shape:
        raise ValueError(f"Expected RGB array with one dimension of size 3, got {arr.shape}.")

    if arr.shape[0] == 3:
        r, g, b = arr[0], arr[1], arr[2]
    else:
        r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]

    if np.max([r.max(), g.max(), b.max()]) > 1.0:
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    r = r.reshape(1, -1)
    g = g.reshape(1, -1)
    b = b.reshape(1, -1)

    rgb_stack = np.vstack([r, g, b])

    mat = np.array(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]
    )
    xyz = mat @ rgb_stack

    x = xyz[0, :] / 0.950456
    y = xyz[1, :]
    z = xyz[2, :] / 1.088754

    T = 0.008856

    def f(t: np.ndarray) -> np.ndarray:
        mask = t > T
        ft = np.empty_like(t)
        ft[mask] = np.cbrt(t[mask])
        ft[~mask] = 7.787 * t[~mask] + 16.0 / 116.0
        return ft

    fx = f(x)
    fy = f(y)
    fz = f(z)

    y3 = np.cbrt(y)
    yt = y > T
    L = np.empty_like(y)
    L[yt] = 116.0 * y3[yt] - 16.0
    L[~yt] = 903.3 * y[~yt]

    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)

    L = L.reshape(1, -1)
    a = a.reshape(1, -1)
    b_lab = b_lab.reshape(1, -1)

    lab = np.vstack([L, a, b_lab])

    lab = lab / np.array([[100.0], [128.0], [128.0]])
    return lab


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB colors to HSV using matplotlib, matching MATLAB rgb2hsv behavior.

    Args:
        rgb: Array of shape (3, num_colors) or (num_colors, 3).

    Returns:
        HSV array with shape (3, num_colors) in [0, 1].
    """
    arr = np.asarray(rgb, dtype=float)
    if arr.ndim != 2 or 3 not in arr.shape:
        raise ValueError(f"Expected RGB array with one dimension of size 3, got {arr.shape}.")

    if arr.shape[0] == 3:
        arr = arr.T

    if arr.max() > 1.0:
        arr = arr / 255.0

    hsv = mpl_rgb_to_hsv(arr)
    return hsv.T


def _get_plane_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Python equivalent of getPlaneFeatures.m.

    Args:
        X: Array of shape (num_points, 3) representing coordinates.

    Returns:
        normal: Plane normal vector (3,).
        pct_explained: Variance explained by each principal component (3,).
        mean_x: Mean of points (3,).
        sse: Sum of squared errors to the fitted plane.
    """
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected X with shape (num_points, 3), got {X.shape}.")

    mean_x = X.mean(axis=0, keepdims=True)
    Xc = X - mean_x

    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PC = Vt.T

    normal = PC[:, 2]
    if normal[0] < 0:
        normal = -normal

    roots = S**2
    if roots.sum() == 0:
        pct_explained = np.zeros(3, dtype=float)
    else:
        pct_explained = roots / roots.sum()

    mean_x_vec = mean_x.reshape(-1)
    error = np.abs((X - mean_x_vec) @ normal)
    sse = float((error**2).sum())
    return normal, pct_explained, mean_x_vec, sse


def _load_hue_probs_and_mapping(
    base_path: str = "data/data",
) -> Tuple[object, CubicSpline]:
    """
    Load hueProbsRGB.mat (hueProbs struct) and kulerX.mat (x vector) and build
    the hue remapping spline, matching the MATLAB code:
        load hueProbsRGB
        load kulerX
        y = (0:360)./360;
        mapping = spline(x, y);
    """
    hue_mat = loadmat(f"{base_path}/hueProbsRGB.mat", struct_as_record=False, squeeze_me=True)
    if "hueProbs" not in hue_mat:
        raise KeyError("Expected variable 'hueProbs' in hueProbsRGB.mat.")
    hue_probs = hue_mat["hueProbs"]

    kuler_mat = loadmat(f"{base_path}/kulerX.mat", struct_as_record=False, squeeze_me=True)
    if "x" not in kuler_mat:
        raise KeyError("Expected variable 'x' in kulerX.mat.")
    x = np.asarray(kuler_mat["x"], dtype=float).reshape(-1)

    # In the original MATLAB code the call is:
    #   y = (0:360)./360;
    #   mapping = spline(x, y);
    # MATLAB's `spline` is more permissive about nearly‑monotone `x` than
    # SciPy's CubicSpline, which requires a strictly increasing sequence.
    # The kulerX.mat values are almost, but not perfectly, monotone; to
    # satisfy CubicSpline we sort (x, y) by x before constructing the spline.
    y = np.linspace(0.0, 1.0, 361)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    mapping = CubicSpline(x_sorted, y_sorted)
    return hue_probs, mapping


def _compute_color_spaces(rgb: np.ndarray, mapping: CubicSpline) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python equivalent of getColorSpaces.m (including hue remapping).

    Args:
        rgb: Array of shape (3, num_colors).
        mapping: CubicSpline used to remap hue.

    Returns:
        hsv, lab, chsv arrays (each 3 x num_colors).
    """
    hsv = _rgb_to_hsv(rgb)
    lab = _rgb_to_lab(rgb)

    hsv_remap = hsv.copy()
    hsv_remap[0, :] = mapping(hsv_remap[0, :])

    angles = 2.0 * np.pi * hsv_remap[0, :]
    chsv = np.vstack(
        [
            hsv_remap[1, :] * np.cos(angles),
            -hsv_remap[1, :] * np.sin(angles),
            hsv_remap[2, :],
        ]
    )

    return hsv, lab, chsv


def _get_basic_stats(x: np.ndarray, add_log: bool = True) -> np.ndarray:
    """
    Python version of getBasicStats.m.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(8, dtype=float)

    log_x = np.log(x + 1e-6)
    features = np.array(
        [
            x.mean(),
            x.std(),
            x.min(),
            x.max(),
            log_x.mean(),
            log_x.std(),
            log_x.min(),
            log_x.max(),
        ],
        dtype=float,
    )
    features[~np.isfinite(features)] = 0.0
    return features


def _circ_vmpdf(alpha: np.ndarray, thetahat: float, kappa: float) -> np.ndarray:
    """
    Python implementation of circ_vmpdf.m using scipy.special.i0.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    C = 1.0 / (2.0 * np.pi * i0(kappa))
    return C * np.exp(kappa * np.cos(alpha - thetahat))


def _get_hue_prob_features(hsv: np.ndarray, sat_val_thresh: float, hue_probs: object) -> np.ndarray:
    """
    Python equivalent of getHueProbFeatures.m.
    """
    hsv = np.asarray(hsv, dtype=float)
    if hsv.shape[0] != 3:
        raise ValueError(f"Expected hsv with shape (3, num_colors), got {hsv.shape}.")

    select_colors = np.min(hsv[1:3, :], axis=0) >= sat_val_thresh

    scale = np.array([[359.0], [100.0], [100.0]])
    hsv2 = np.round(hsv * scale).astype(int) + 1
    vis_hues = hsv2[0, select_colors]

    hue_joint_list: List[float] = []
    hue_adj_list: List[float] = []

    if vis_hues.size > 0:
        hue_prob_arr = np.asarray(hue_probs.hueProb)
        hue_joint_arr = np.asarray(hue_probs.hueJoint)
        hue_adj_arr = np.asarray(hue_probs.hueAdjacency)

        for h1_idx in range(vis_hues.size):
            for h2_idx in range(h1_idx, vis_hues.size):
                i1 = vis_hues[h1_idx] - 1
                i2 = vis_hues[h2_idx] - 1
                hue_joint_list.append(float(hue_joint_arr[i2, i1]))

        for h1_idx in range(vis_hues.size - 1):
            i1 = vis_hues[h1_idx] - 1
            i2 = vis_hues[h1_idx + 1] - 1
            hue_adj_list.append(float(hue_adj_arr[i1, i2]))

        hue_prob_features = _get_basic_stats(hue_prob_arr[vis_hues - 1], add_log=True)
        hue_joint_prob_features = _get_basic_stats(np.asarray(hue_joint_list), add_log=True)
        hue_adj_prob_features = _get_basic_stats(np.asarray(hue_adj_list), add_log=True)
    else:
        hue_prob_features = np.zeros(8, dtype=float)
        hue_joint_prob_features = np.zeros(8, dtype=float)
        hue_adj_prob_features = np.zeros(8, dtype=float)

    alpha = np.linspace(0.0, 2.0 * np.pi, 361)[:-1]
    p_mix = 0.001 * np.ones_like(alpha)
    for h in vis_hues:
        p_mix = p_mix + _circ_vmpdf(alpha, float(h) * 2.0 * np.pi / 360.0, 2.0 * np.pi)

    if vis_hues.size != 0:
        p_mix = p_mix / p_mix.sum()
        entropy = float(-(p_mix * np.log(p_mix + 1e-12)).sum())
    else:
        entropy = 5.9

    return np.concatenate([hue_prob_features, hue_joint_prob_features, hue_adj_prob_features, [entropy]])


def _build_feature_names_for_space(
    name: str, n_colors: int
) -> Dict[str, List[str]]:
    """
    Construct feature-name lists mirroring the MATLAB createFeaturesFromData.m logic.
    """
    names: Dict[str, List[str]] = {}

    col_labels: List[str] = []
    for j in range(1, n_colors + 1):
        for d in range(1, 4):
            col_labels.append(f"{name}-D{d}-C{j}")
    names[f"{name}Col"] = col_labels

    sorted_col_labels: List[str] = []
    for j in range(1, n_colors + 1):
        for d in range(1, 4):
            sorted_col_labels.append(f"{name}Sorted-D{d}-C{j}")
    names[f"{name}SortedCol"] = sorted_col_labels

    diff_labels: List[str] = []
    for d in range(1, 4):
        for j in range(1, n_colors):
            diff_labels.append(f"{name}Diff-D{d}-C{j}")
    names[f"{name}Diff"] = diff_labels

    sorted_diff_labels: List[str] = []
    for d in range(1, 4):
        for j in range(1, n_colors):
            sorted_diff_labels.append(f"{name}SortedDiff-D{d}-C{j}")
    names[f"{name}SortedDiff"] = sorted_diff_labels

    names[f"{name}Mean"] = [f"{name}Mean-D{d}" for d in range(1, 4)]
    names[f"{name}StdDev"] = [f"{name}StdDev-D{d}" for d in range(1, 4)]
    names[f"{name}Median"] = [f"{name}Median-D{d}" for d in range(1, 4)]
    names[f"{name}Max"] = [f"{name}Max-D{d}" for d in range(1, 4)]
    names[f"{name}Min"] = [f"{name}Min-D{d}" for d in range(1, 4)]
    names[f"{name}MaxMinDiff"] = [f"{name}MaxMinDiff-D{d}" for d in range(1, 4)]

    if name != "hsv":
        names[f"{name}Plane"] = [
            f"{name}PlaneNormal1",
            f"{name}PlaneNormal2",
            f"{name}PlaneNormal3",
            f"{name}PlaneVariance-D1",
            f"{name}PlaneVariance-D2",
            f"{name}PlaneVariance-D3",
            f"{name}SSE",
        ]
    else:
        prefix_names = []
        for prefix in [f"{name}HueProb", f"{name}HueJointProb", f"{name}HueAdjProb"]:
            prefix_names.extend(
                [
                    f"{prefix}Mean",
                    f"{prefix}StdDev",
                    f"{prefix}Min",
                    f"{prefix}Max",
                    f"{prefix}LogMean",
                    f"{prefix}LogStdDev",
                    f"{prefix}LogMin",
                    f"{prefix}LogMax",
                ]
            )
        prefix_names.append(f"{name}Entropy")
        names[f"{name}HueProb"] = prefix_names

    return names


def create_features_from_data_array(
    data: np.ndarray,
    max_features: Optional[int] = None,
    sat_val_thresh: float = 0.2,
    aux_base_path: str = "data/data",
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]], int, np.ndarray, np.ndarray]:
    """
    Python translation of createFeaturesFromData.m for use on mturkData.mat `data`.

    This ports the RGB/Lab/HSV/CHSV feature families (colors, sorted colors,
    diffs, sorted diffs, basic stats, plane features), and hue-probability
    features using hueProbsRGB.mat and kulerX.mat.

    Args:
        data: Array of shape (N, 5, 3) with palette colors.
        max_features: Optional cap on number of rows to process.
        sat_val_thresh: Threshold used in the circular hue-difference logic.
        aux_base_path: Directory containing hueProbsRGB.mat and kulerX.mat.

    Returns:
        all_features: dict mapping feature-group name -> (num_rows, num_features_in_group).
        feature_names: dict mapping feature-group name -> list of column labels.
        num_themes: number of rows actually processed.
        rgbs: flattened RGB values per row (num_rows, 15).
        labs: flattened Lab values per row (num_rows, 15).
    """
    if data.ndim != 3 or data.shape[1] != 5 or data.shape[2] != 3:
        raise ValueError(f"Expected data with shape (N, 5, 3), got {data.shape}.")

    hue_probs, mapping = _load_hue_probs_and_mapping(aux_base_path)

    n_samples = data.shape[0]
    num_themes = n_samples if max_features is None else min(int(max_features), n_samples)
    n_colors = data.shape[1]

    rgbs = np.zeros((num_themes, 3 * n_colors), dtype=float)
    labs = np.zeros((num_themes, 3 * n_colors), dtype=float)

    rgb_mats: List[np.ndarray] = []
    hsv_mats: List[np.ndarray] = []
    lab_mats: List[np.ndarray] = []
    chsv_mats: List[np.ndarray] = []
    num_colors_list: List[int] = []

    for i in range(num_themes):
        rgb_full = data[i].T
        mask_nonneg = rgb_full[0, :] >= 0
        num_colors_i = int(mask_nonneg.sum())
        if num_colors_i == 0:
            num_colors_i = rgb_full.shape[1]
        rgb = rgb_full[:, :num_colors_i]

        hsv, lab, chsv = _compute_color_spaces(rgb, mapping)

        rgb_mats.append(rgb)
        hsv_mats.append(hsv)
        lab_mats.append(lab)
        chsv_mats.append(chsv)
        num_colors_list.append(num_colors_i)

        rgbs[i, : 3 * num_colors_i] = rgb.reshape(-1, order="F")
        labs[i, : 3 * num_colors_i] = lab.reshape(-1, order="F")

    all_features: Dict[str, np.ndarray] = {}
    feature_names: Dict[str, List[str]] = {}

    dummy_hsv = np.random.rand(3, n_colors)
    dummy_hue_feats = _get_hue_prob_features(dummy_hsv, sat_val_thresh, hue_probs)
    hue_feat_len = dummy_hue_feats.shape[0]

    for name in ["chsv", "lab", "hsv", "rgb"]:
        feature_names.update(_build_feature_names_for_space(name, n_colors))

        color_arr = np.zeros((num_themes, 3 * n_colors), dtype=float)
        sorted_col_arr = np.zeros((num_themes, 3 * n_colors), dtype=float)

        diff_arr = np.zeros((num_themes, 3 * (n_colors - 1)), dtype=float)
        sorted_diff_arr = np.zeros((num_themes, 3 * (n_colors - 1)), dtype=float)

        means_arr = np.zeros((num_themes, 3), dtype=float)
        stddevs_arr = np.zeros((num_themes, 3), dtype=float)
        medians_arr = np.zeros((num_themes, 3), dtype=float)
        mins_arr = np.zeros((num_themes, 3), dtype=float)
        maxs_arr = np.zeros((num_themes, 3), dtype=float)
        max_min_diff_arr = np.zeros((num_themes, 3), dtype=float)

        plane_arr = np.zeros((num_themes, 7), dtype=float)
        hue_prob_arr = -99.0 * np.ones((num_themes, hue_feat_len), dtype=float)

        for i in range(num_themes):
            num_colors_i = num_colors_list[i]
            if name == "chsv":
                col = chsv_mats[i]
            elif name == "lab":
                col = lab_mats[i]
            elif name == "hsv":
                col = hsv_mats[i]
            else:
                col = rgb_mats[i]

            col = col[:, :num_colors_i]
            color_arr[i, : 3 * num_colors_i] = col.reshape(-1, order="F")

            diffs = np.zeros((3, max(num_colors_i - 1, 1)), dtype=float)
            for j in range(1, num_colors_i):
                if name == "hsv":
                    hsv_curr = hsv_mats[i][:, :num_colors_i]
                    min_sat_val = min(
                        np.min(hsv_curr[1, j - 1 : j + 1]),
                        np.min(hsv_curr[2, j - 1 : j + 1]),
                    )
                    if min_sat_val >= sat_val_thresh:
                        pts = np.sort([col[0, j], col[0, j - 1]])
                        diffs[0, j - 1] = min(
                            pts[1] - pts[0],
                            1.0 - (pts[1] - pts[0]),
                        )
                else:
                    diffs[0, j - 1] = col[0, j] - col[0, j - 1]

                diffs[1, j - 1] = col[1, j] - col[1, j - 1]
                diffs[2, j - 1] = col[2, j] - col[2, j - 1]

            diff_arr[i, : 3 * (num_colors_i - 1)] = np.concatenate(
                [
                    diffs[0, : num_colors_i - 1],
                    diffs[1, : num_colors_i - 1],
                    diffs[2, : num_colors_i - 1],
                ]
            )

            num_diffs = num_colors_i - 1
            if num_diffs > 0:
                sorted_diff_arr[i, :num_diffs] = np.sort(diffs[0, :num_diffs])[::-1]
                sorted_diff_arr[i, num_diffs : 2 * num_diffs] = np.sort(diffs[1, :num_diffs])[::-1]
                sorted_diff_arr[i, 2 * num_diffs : 3 * num_diffs] = np.sort(diffs[2, :num_diffs])[::-1]

            means_arr[i, :] = np.mean(col, axis=1)
            stddevs_arr[i, :] = np.std(col, axis=1, ddof=0)
            medians_arr[i, :] = np.median(col, axis=1)
            mins_arr[i, :] = np.min(col, axis=1)
            maxs_arr[i, :] = np.max(col, axis=1)
            max_min_diff_arr[i, :] = maxs_arr[i, :] - mins_arr[i, :]

            if name != "hsv":
                X = col.T
                normal, pct_explained, _, sse = _get_plane_features(X)
                plane_arr[i, 0:3] = normal
                plane_arr[i, 3:6] = pct_explained
                plane_arr[i, 6] = sse
            else:
                hue_prob_arr[i, :] = _get_hue_prob_features(col, sat_val_thresh, hue_probs)

            order = np.argsort(col[2, :])
            col_sorted = col[:, order]
            sorted_col_arr[i, : 3 * num_colors_i] = col_sorted.reshape(-1, order="F")

        all_features[f"{name}Col"] = color_arr
        all_features[f"{name}SortedCol"] = sorted_col_arr
        all_features[f"{name}Diff"] = diff_arr
        all_features[f"{name}SortedDiff"] = sorted_diff_arr
        all_features[f"{name}Mean"] = means_arr
        all_features[f"{name}StdDev"] = stddevs_arr
        all_features[f"{name}Median"] = medians_arr
        all_features[f"{name}Max"] = maxs_arr
        all_features[f"{name}Min"] = mins_arr
        all_features[f"{name}MaxMinDiff"] = max_min_diff_arr

        if name != "hsv":
            all_features[f"{name}Plane"] = plane_arr
        else:
            for col_idx in range(hue_prob_arr.shape[1]):
                col_vals = hue_prob_arr[:, col_idx]
                mask_neg = col_vals == -99.0
                if np.any(~mask_neg):
                    col_max = float(col_vals[~mask_neg].max())
                    col_vals[mask_neg] = col_max + 0.0001
                    hue_prob_arr[:, col_idx] = col_vals
            all_features[f"{name}HueProb"] = hue_prob_arr

    return all_features, feature_names, num_themes, rgbs, labs


def add_palette_features_to_df(
    df: pd.DataFrame, max_rows: Optional[int] = None, aux_base_path: str = "data/data"
) -> pd.DataFrame:
    """
    Compute color-palette features (Python port of createFeaturesFromData.m)
    for each row of the MTurk dataset and return a new DataFrame with features
    appended as additional columns.

    Args:
        df: DataFrame created by loadData, with color1_r ... color5_b columns.
        max_rows: Optional limit on number of rows to process.
        aux_base_path: Directory containing hueProbsRGB.mat and kulerX.mat.

    Returns:
        New DataFrame where each original row also has the engineered
        color features as extra columns.
    """
    color_blocks: List[np.ndarray] = []
    for idx in range(1, 6):
        cols = [f"color{idx}_r", f"color{idx}_g", f"color{idx}_b"]
        if not all(c in df.columns for c in cols):
            raise KeyError(f"Expected color columns {cols} in DataFrame.")
        color_blocks.append(df[cols].to_numpy())

    data_arr = np.stack(color_blocks, axis=1)

    all_features, feature_names, num_themes, _, _ = create_features_from_data_array(
        data_arr, max_features=max_rows, aux_base_path=aux_base_path
    )

    feature_cols: Dict[str, np.ndarray] = {}
    for group_name, arr in all_features.items():
        names = feature_names.get(group_name)
        if names is None or len(names) != arr.shape[1]:
            names = [f"{group_name}_{i}" for i in range(arr.shape[1])]
        for col_idx, col_name in enumerate(names):
            if col_name in feature_cols:
                col_name = f"{col_name}_{group_name}"
            feature_cols[col_name] = arr[:, col_idx]

    feats_df = pd.DataFrame(feature_cols)

    base_df = df.iloc[:num_themes].reset_index(drop=True)
    feats_df = feats_df.iloc[:num_themes].reset_index(drop=True)
    return pd.concat([base_df, feats_df], axis=1)


def select_top_mturk_features(
    df: pd.DataFrame,
    weights_csv_path: str = "data/weights.csv",
    k: int = TOP_K_MTURK_FEATURES,
) -> pd.DataFrame:
    """
    Keep only the top-k Mturk-weighted feature columns (plus core metadata).

    The weights file is expected to have columns:
        Feature, Kuler, ColorLovers, Mturk
    and feature names that match the engineered feature column names
    (e.g., 'chsv-D1-C1', 'labMean-D1', 'hsvHueProbMean', ...).
    """
    if k is None or k <= 0:
        return df

    try:
        weights_df = pd.read_csv(weights_csv_path)
    except FileNotFoundError:
        # If no weights file is available, return the original dataframe.
        return df

    if "Feature" not in weights_df.columns or "Mturk" not in weights_df.columns:
        return df

    weights_df["Feature"] = weights_df["Feature"].astype(str).str.strip()
    weights_df = weights_df.dropna(subset=["Mturk"])

    if weights_df.empty:
        return df

    weights_df["abs_weight"] = weights_df["Mturk"].abs()
    # Keep only features that actually exist as columns in df
    weights_df = weights_df[weights_df["Feature"].isin(df.columns)]
    if weights_df.empty:
        return df

    top_features = (
        weights_df.sort_values("abs_weight", ascending=False)["Feature"].head(k).tolist()
    )

    # Always keep core metadata/target columns if present
    core_cols = [c for c in ["ids", "names", "targets", "userNormalizedTargets"] if c in df.columns]
    selected_cols = core_cols + [c for c in top_features if c not in core_cols]

    # Some rows may have been truncated during feature creation; keep all rows, only trim columns.
    return df.loc[:, selected_cols]


def prune_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Drop one feature from any pair whose absolute correlation exceeds `threshold`.

    Core metadata/target columns ('ids', 'names', 'targets', 'userNormalizedTargets')
    are always retained and are not candidates for dropping.
    """
    core_cols = [c for c in ["ids", "names", "targets", "userNormalizedTargets"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in core_cols]

    if len(feature_cols) <= 1:
        print("\nFeature pruning: fewer than two feature columns; nothing to drop.")
        return df

    # Compute absolute correlation matrix for feature columns only
    corr = df[feature_cols].corr().abs()
    # Use the upper triangle to avoid duplicate pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    drop_pairs: List[Tuple[str, str, float]] = []  # (dropped, kept, corr)

    for col in feature_cols:
        if col in to_drop:
            continue
        # Find features highly correlated with `col`
        high_corr = upper.index[upper[col] > threshold].tolist()
        for other in high_corr:
            if other not in to_drop:
                to_drop.add(other)
                drop_pairs.append((other, col, float(upper.loc[other, col])))

    if not drop_pairs:
        print(f"\nFeature pruning: no pairs with |corr| > {threshold}; nothing dropped.")
        return df

    print(f"\nFeature pruning (|corr| > {threshold}):")
    for dropped, kept, corr_val in drop_pairs:
        print(f"  Dropped '{dropped}' (kept '{kept}'), corr={corr_val:.3f}")

    pruned_df = df.drop(columns=list(to_drop))
    print(f"Feature pruning: dropped {len(to_drop)} of {len(feature_cols)} feature columns.")
    return pruned_df


def trainRatingBaseline(
    dfWithFeatures: pd.DataFrame,
    featureCols: List[str],
    testSize: float = 0.2,
    randomState: int = 59,
):
    """
    Train a rating-based baseline model to predict userNormalizedTargets from palette features.

    Steps:
        - Split palettes into train/test.
        - Scale features with StandardScaler.
        - Train a LASSO (L1) regression model.
        - Compute RMSE on the test set.

    Returns:
        model: fitted sklearn Lasso instance.
        scaler: fitted StandardScaler instance.
        X_train_scaled, X_test_scaled, y_train, y_test: train/test splits (features are scaled).
        train_rmse: float, RMSE on the training set.
        test_rmse: float, RMSE on the test set.
    """
    if "userNormalizedTargets" not in dfWithFeatures.columns:
        raise KeyError("Column 'userNormalizedTargets' not found in dfWithFeatures.")

    missing = [c for c in featureCols if c not in dfWithFeatures.columns]
    if missing:
        raise KeyError(f"The following feature columns are missing in dfWithFeatures: {missing}")

    # Extract features and target
    X = dfWithFeatures[featureCols].to_numpy(dtype=float)
    y = dfWithFeatures["userNormalizedTargets"].to_numpy(dtype=float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, random_state=randomState
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LASSO regression baseline
    model = Lasso(alpha=0.001, max_iter=10000, random_state=randomState)
    model.fit(X_train_scaled, y_train)

    # Evaluate on train and test sets (RMSE)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

    return (
        model,
        scaler,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        train_rmse,
        test_rmse,
    )


def generateSyntheticPairs(
    train: pd.DataFrame,
    nPairs: int = 100000,
    delta: float = 0.2,
    randomState: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly sample palette pairs from a training dataframe and generate preference labels.

    Pairs where |userNormalizedTargets_i - userNormalizedTargets_j| < delta are skipped.

    Args:
        train: DataFrame containing userNormalizedTargets and palette feature columns.
        nPairs: Target number of synthetic pairs to generate.
        delta: Minimum absolute difference in userNormalizedTargets to keep a pair.
        randomState: Seed for the random number generator.

    Returns:
        Xi: np.ndarray of shape (n_kept, n_features), features for item i.
        Xj: np.ndarray of shape (n_kept, n_features), features for item j.
        yPairs: np.ndarray of shape (n_kept,), with labels:
                1 if userNormalizedTargets_i > userNormalizedTargets_j else 0.
    """
    if "userNormalizedTargets" not in train.columns:
        raise KeyError("Column 'userNormalizedTargets' not found in training dataframe.")

    feature_cols = [
        c for c in train.columns if c not in ["ids", "names", "targets", "userNormalizedTargets"]
    ]
    if not feature_cols:
        raise ValueError("No feature columns found in training dataframe.")

    n = len(train)
    if n < 2:
        raise ValueError("Need at least two palettes in training data to form pairs.")

    X_all = train[feature_cols].to_numpy(dtype=float)
    y_all = train["userNormalizedTargets"].to_numpy(dtype=float)

    rng = np.random.default_rng(randomState)

    Xi_list: List[np.ndarray] = []
    Xj_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    total_sampled = 0
    target_pairs = int(nPairs)
    # Oversample in batches to efficiently filter by delta.
    batch_size = min(max(target_pairs * 2, 1000), n * n)

    while sum(len(b) for b in y_list) < target_pairs and total_sampled < target_pairs * 20:
        size = batch_size
        idx_i = rng.integers(0, n, size=size)
        idx_j = rng.integers(0, n, size=size)

        # Exclude self-pairs
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        if idx_i.size == 0:
            total_sampled += size
            continue

        diff = y_all[idx_i] - y_all[idx_j]
        keep = np.abs(diff) >= delta
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]
        diff = diff[keep]

        total_sampled += size

        if idx_i.size == 0:
            continue

        remaining = target_pairs - sum(len(b) for b in y_list)
        if remaining <= 0:
            break

        if idx_i.size > remaining:
            idx_i = idx_i[:remaining]
            idx_j = idx_j[:remaining]
            diff = diff[:remaining]

        Xi_list.append(X_all[idx_i])
        Xj_list.append(X_all[idx_j])
        y_list.append((diff > 0).astype(int))

    if not y_list:
        print(
            f"\nSynthetic pair generation: no valid pairs found with "
            f"|delta| >= {delta}. Returning empty arrays."
        )
        return (
            np.empty((0, len(feature_cols)), dtype=float),
            np.empty((0, len(feature_cols)), dtype=float),
            np.empty((0,), dtype=int),
        )

    Xi = np.vstack(Xi_list)
    Xj = np.vstack(Xj_list)
    yPairs = np.concatenate(y_list)

    n_kept = Xi.shape[0]

    print("\nSynthetic pair generation summary:")
    print(f"  Requested pairs: {target_pairs}")
    print(f"  Total candidate pairs sampled: {total_sampled}")
    print(f"  Pairs kept after |delta| >= {delta}: {n_kept}")
    if n_kept < target_pairs:
        print(
            "  Note: fewer pairs than requested were generated; "
            "consider lowering delta or increasing training data size."
        )

    if n_kept > 0:
        prop_1 = float(yPairs.mean())
        prop_0 = 1.0 - prop_1
        print(f"  Proportion of label 1: {prop_1:.3f}")
        print(f"  Proportion of label 0: {prop_0:.3f}")

    return Xi, Xj, yPairs


def trainBtLogisticModel(
    Xi: np.ndarray,
    Xj: np.ndarray,
    yPairs: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    randomState: int = 0,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train a Bradley–Terry style logistic regression model on synthetic pairwise data.

    The model operates on feature differences (Xi - Xj) to predict yPairs.

    Args:
        Xi: Array of shape (n_pairs, n_features), features for item i.
        Xj: Array of shape (n_pairs, n_features), features for item j.
        yPairs: Array of shape (n_pairs,), with labels 1 if i is preferred to j, else 0.
        C: Inverse of regularization strength for LogisticRegression (L2 penalty).
        max_iter: Maximum number of iterations for the solver.
        randomState: Random seed for the LogisticRegression model.

    Returns:
        model: Fitted sklearn LogisticRegression instance.
        scaler: Fitted StandardScaler instance used on (Xi - Xj).
    """
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)
    yPairs = np.asarray(yPairs).astype(int)

    if Xi.shape != Xj.shape:
        raise ValueError(f"Xi and Xj must have the same shape, got {Xi.shape} vs {Xj.shape}.")
    if Xi.shape[0] != yPairs.shape[0]:
        raise ValueError(
            f"Number of pairs in Xi/Xj and yPairs must match, "
            f"got {Xi.shape[0]} vs {yPairs.shape[0]}."
        )
    if Xi.shape[0] == 0:
        raise ValueError("No pairwise data provided: Xi has zero rows.")

    # Construct feature differences for Bradley–Terry style modeling
    X_diff = Xi - Xj

    # Standardize features
    scaler = StandardScaler()
    X_diff_scaled = scaler.fit_transform(X_diff)

    # L2-regularized logistic regression
    model = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=randomState,
    )
    model.fit(X_diff_scaled, yPairs)

    return model, scaler


def evaluateBtModelOnTestPairs(
    model: LogisticRegression,
    scaler: StandardScaler,
    XTest: np.ndarray,
    yTest: np.ndarray,
    nPairs: int = 20000,
    delta: float = 0.2,
) -> Tuple[float, float]:
    """
    Evaluate a Bradley–Terry logistic model on synthetic test pairs.

    Randomly samples pairs from the test set, filters out pairs with
    |y_i - y_j| < delta, and computes:
        - Pairwise accuracy (fraction of correctly ordered pairs).
        - ROC AUC based on predicted pairwise probabilities.

    Args:
        model: Fitted LogisticRegression model from trainBtLogisticModel.
        scaler: StandardScaler fitted on (Xi - Xj) during training.
        XTest: Array of shape (n_samples, n_features) with test palette features.
        yTest: Array of shape (n_samples,) with test userNormalizedTargets.
        nPairs: Target number of test pairs to sample.
        delta: Minimum absolute rating difference required to keep a pair.

    Returns:
        pair_accuracy: float, pairwise accuracy over sampled pairs.
        pair_auc: float, ROC AUC over sampled pairs (np.nan if undefined).
    """
    XTest = np.asarray(XTest, dtype=float)
    yTest = np.asarray(yTest, dtype=float)

    if XTest.ndim != 2:
        raise ValueError(f"XTest must be 2D, got shape {XTest.shape}.")
    if XTest.shape[0] != yTest.shape[0]:
        raise ValueError(
            f"Number of samples in XTest and yTest must match, "
            f"got {XTest.shape[0]} vs {yTest.shape[0]}."
        )
    if XTest.shape[0] < 2:
        raise ValueError("Need at least two test samples to form pairs.")

    n = XTest.shape[0]
    target_pairs = int(nPairs)
    rng = np.random.default_rng(0)

    Xdiff_list: List[np.ndarray] = []
    ypair_list: List[np.ndarray] = []
    total_sampled = 0

    # Oversample in batches so we can filter by delta efficiently
    batch_size = min(max(target_pairs * 2, 1000), n * n)

    while sum(len(b) for b in ypair_list) < target_pairs and total_sampled < target_pairs * 20:
        size = batch_size
        idx_i = rng.integers(0, n, size=size)
        idx_j = rng.integers(0, n, size=size)

        # Exclude self-pairs
        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]

        if idx_i.size == 0:
            total_sampled += size
            continue

        diff_y = yTest[idx_i] - yTest[idx_j]
        keep = np.abs(diff_y) >= delta
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]
        diff_y = diff_y[keep]

        total_sampled += size

        if idx_i.size == 0:
            continue

        remaining = target_pairs - sum(len(b) for b in ypair_list)
        if remaining <= 0:
            break

        if idx_i.size > remaining:
            idx_i = idx_i[:remaining]
            idx_j = idx_j[:remaining]
            diff_y = diff_y[:remaining]

        Xdiff_list.append(XTest[idx_i] - XTest[idx_j])
        ypair_list.append((diff_y > 0).astype(int))

    if not ypair_list:
        print(
            f"\nBT test evaluation: no valid pairs found with "
            f"|delta| >= {delta}. Returning NaN metrics."
        )
        return float("nan"), float("nan")

    X_diff = np.vstack(Xdiff_list)
    yPairs = np.concatenate(ypair_list)
    n_kept = X_diff.shape[0]

    X_diff_scaled = scaler.transform(X_diff)
    proba = model.predict_proba(X_diff_scaled)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    pair_accuracy = float((y_pred == yPairs).mean())

    if yPairs.min() == yPairs.max():
        pair_auc = float("nan")
        print(
            "\nBT test evaluation: only one class present in yPairs; "
            "ROC AUC is undefined (set to NaN)."
        )
    else:
        pair_auc = float(roc_auc_score(yPairs, proba))

    print("\nBT test pair evaluation summary:")
    print(f"  Requested test pairs: {target_pairs}")
    print(f"  Total candidate test pairs sampled: {total_sampled}")
    print(f"  Test pairs kept after |delta| >= {delta}: {n_kept}")
    print(f"  Pairwise accuracy: {pair_accuracy:.4f}")
    if np.isfinite(pair_auc):
        print(f"  Pairwise ROC AUC: {pair_auc:.4f}")
    else:
        print("  Pairwise ROC AUC: NaN (undefined due to single-class labels).")

    return pair_accuracy, pair_auc


def summarizeRatings(df: pd.DataFrame) -> None:
    """
    Print summary statistics and plot a histogram for targets.
    """
    if "targets" not in df.columns:
        raise KeyError("Column 'targets' not found in DataFrame.")

    meanRating = df["targets"].mean()
    stdRating = df["targets"].std()

    print("Mean of targets:", meanRating)
    print("Std of targets:", stdRating)

    plt.figure(figsize=(6, 4))
    plt.hist(df["targets"], bins=20, edgecolor="black")
    plt.xlabel("targets")
    plt.ylabel("Frequency")
    plt.title("Distribution of targets")
    plt.tight_layout()
    # Save to file so the plot is available even in non-interactive environments.
    plt.savefig("targets_hist.png")
    # plt.show()


def summarizeUserNormalizedTargets(df: pd.DataFrame) -> None:
    """
    Print summary statistics and plot a histogram for userNormalizedTargets.
    """
    colName = "userNormalizedTargets"
    if colName not in df.columns:
        raise KeyError(f"Column '{colName}' not found in DataFrame.")

    meanVal = df[colName].mean()
    stdVal = df[colName].std()

    print(f"Mean of {colName}:", meanVal)
    print(f"Std of {colName}:", stdVal)

    plt.figure(figsize=(6, 4))
    plt.hist(df[colName], bins=20, edgecolor="black")
    plt.xlabel(colName)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {colName}")
    plt.tight_layout()
    plt.savefig("user_normalized_targets_hist.png")
    # plt.show()


def main() -> None:
    df = loadData()

    print("Base dataset shape:", df.shape)
    print("Base columns:", list(df.columns))
    print("\nBase head:")
    print(df.head())

    print("\nAll base columns describe():")
    print(df.describe(include="all"))

    if "targets" in df.columns:
        print("\nBase targets describe():")
        print(df["targets"].describe())
    else:
        print("\nColumn 'targets' not found; please adjust loader.")

    summarizeRatings(df)

    if "userNormalizedTargets" in df.columns:
        print("\nBase userNormalizedTargets describe():")
        print(df["userNormalizedTargets"].describe())
        summarizeUserNormalizedTargets(df)

    # Compute and attach all color features (Python port of createFeaturesFromData.m)
    df_with_features = add_palette_features_to_df(df, aux_base_path="data/data")

    # Restrict to the top-k Mturk-weighted features
    df_top = select_top_mturk_features(df_with_features, weights_csv_path="data/weights.csv")

    # Prune highly correlated features among the selected feature columns
    df_final = prune_highly_correlated_features(df_top, threshold=0.95)

    print("\nFinal dataframe after Mturk selection and correlation pruning:")
    print("Final shape:", df_final.shape)
    print("Number of columns:", len(df_final.columns))
    print("Final columns:")
    print(list(df_final.columns))
    print("\nFinal head:")
    print(df_final.head())

    # Train and evaluate a rating-based baseline model on the final feature set
    feature_cols = [
        c for c in df_final.columns if c not in ["ids", "names", "targets", "userNormalizedTargets"]
    ]
    if feature_cols:
        (
            model,
            scaler,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            train_rmse,
            test_rmse,
        ) = trainRatingBaseline(df_final, feature_cols)
        print("\nRating-based baseline (LASSO) performance:")
        print(f"  Number of training examples: {len(y_train)}")
        print(f"  Number of test examples: {len(y_test)}")
        print(f"  Number of feature columns: {len(feature_cols)}")
        print(f"  Train RMSE on userNormalizedTargets: {train_rmse:.4f}")
        print(f"  Test RMSE on userNormalizedTargets: {test_rmse:.4f}")

        # Simple overfitting check: if train error is much lower than test error.
        if train_rmse < 0.9 * test_rmse:
            print(
                "  Overfitting check: WARNING - model may be overfitting "
                "(train RMSE significantly lower than test RMSE)."
            )
        else:
            print(
                "  Overfitting check: no strong evidence of overfitting "
                "based on train/test RMSE."
            )

        # Train and evaluate a Bradley–Terry logistic model using synthetic pairs
        print("\nTraining Bradley–Terry logistic model on synthetic pairs...")
        X_all = df_final[feature_cols].to_numpy(dtype=float)
        y_all = df_final["userNormalizedTargets"].to_numpy(dtype=float)

        X_train_bt, X_test_bt, y_train_bt, y_test_bt = train_test_split(
            X_all, y_all, test_size=0.2, random_state=0
        )

        train_bt_df = pd.DataFrame(X_train_bt, columns=feature_cols)
        train_bt_df["userNormalizedTargets"] = y_train_bt

        Xi_train, Xj_train, yPairs_train = generateSyntheticPairs(
            train_bt_df, nPairs=10000, delta=0.2, randomState=0
        )

        if Xi_train.shape[0] > 0:
            bt_model, bt_scaler = trainBtLogisticModel(Xi_train, Xj_train, yPairs_train)
            pair_acc, pair_auc = evaluateBtModelOnTestPairs(
                bt_model, bt_scaler, X_test_bt, y_test_bt, nPairs=20000, delta=0.2
            )
            print("\nBradley–Terry logistic model performance on test pairs:")
            print(f"  Pairwise accuracy: {pair_acc:.4f}")
            if np.isfinite(pair_auc):
                print(f"  Pairwise ROC AUC: {pair_auc:.4f}")
            else:
                print("  Pairwise ROC AUC: NaN (undefined due to single-class labels).")
        else:
            print(
                "\nBradley–Terry logistic model: no synthetic pairs generated; "
                "skipping training and evaluation."
            )
    else:
        print("\nRating-based baseline: no feature columns available, skipping training.")


if __name__ == "__main__":
    main()
