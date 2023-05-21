"""
Linguistic distance computation using URIEL typological features.

Supports 5 feature types (phylogeny, geography, syntax, phonology, inventory)
and 4 binary distance metrics (Hamming, Jaccard, Inner-product, Anderberg),
resulting in 14 total linguistic distance metrics.

Combined metric from paper:
    d_comb = 0.4 * d_ander-syntax + 0.2 * d_inner-phonology + 0.4 * d_ander-inventory
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional

try:
    import lang2vec.lang2vec as l2v
    LANG2VEC_AVAILABLE = True
except ImportError:
    LANG2VEC_AVAILABLE = False

FEATURE_SETS = {
    "syntax": "syntax_wals|syntax_sswl",
    "phonology": "phonology_wals|phonology_ethnologue",
    "inventory": "inventory_phoible_all|inventory_phoible_gm",
    "fam": "fam",
    "geo": "geo",
}

TYPOLOGICAL_FEATURES = ["syntax", "phonology", "inventory"]
BINARY_METRICS = ["hamming", "jaccard", "inner", "anderberg"]
ALL_DISTANCE_NAMES = (
    [f"{m}-{f}" for f in TYPOLOGICAL_FEATURES for m in BINARY_METRICS]
    + ["fam", "geo"]
)


def _binary_counts(v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, float, float, int]:
    """Compute a, b, c, d, n from 2x2 contingency table for binary feature vectors."""
    mask = np.array([(x != "--" and y != "--") for x, y in zip(v1, v2)])
    if mask.sum() == 0:
        return 0, 0, 0, 0, 0
    v1_m = np.array([float(x) for x, m in zip(v1, mask) if m])
    v2_m = np.array([float(x) for x, m in zip(v2, mask) if m])
    n = len(v1_m)
    a = float(((v1_m == 1) & (v2_m == 1)).sum())
    b = float(((v1_m == 0) & (v2_m == 1)).sum())
    c = float(((v1_m == 1) & (v2_m == 0)).sum())
    d = float(((v1_m == 0) & (v2_m == 0)).sum())
    return a, b, c, d, n


def hamming_distance(v1, v2) -> float:
    """Hamming: b + c (number of differing positions)."""
    a, b, c, d, n = _binary_counts(v1, v2)
    return (b + c) / n if n > 0 else 1.0


def jaccard_distance(v1, v2) -> float:
    """Jaccard: 1 - a/(a+b+c) (complement of Jaccard similarity)."""
    a, b, c, d, n = _binary_counts(v1, v2)
    denom = a + b + c
    return 1.0 - a / denom if denom > 0 else 1.0


def inner_product_distance(v1, v2) -> float:
    """Inner-product: -(a+d) (negative co-occurrence similarity)."""
    a, b, c, d, n = _binary_counts(v1, v2)
    return -(a + d)


def anderberg_distance(v1, v2) -> float:
    """Anderberg: (sigma - sigma') / 2n — captures asymmetric transfer."""
    a, b, c, d, n = _binary_counts(v1, v2)
    if n == 0:
        return 0.0
    sigma = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    sigma_prime = max(a + c, b + d) + max(a + b, c + d)
    return (sigma - sigma_prime) / (2.0 * n)


DISTANCE_FNS = {
    "hamming": hamming_distance,
    "jaccard": jaccard_distance,
    "inner": inner_product_distance,
    "anderberg": anderberg_distance,
}


def get_language_features(lang: str, feature_set: str) -> List:
    """Query URIEL features for a language via lang2vec."""
    if not LANG2VEC_AVAILABLE:
        raise ImportError("lang2vec is required. Install with: pip install lang2vec")
    feats = l2v.get_features(lang, feature_set)
    return feats.get(lang, [])


def compute_pairwise_distances(
    languages: List[str],
    feature_type: str,
    distance_metric: str,
) -> np.ndarray:
    """Compute pairwise (n_lang x n_lang) distance matrix."""
    n = len(languages)
    feature_vecs = {
        lang: get_language_features(lang, FEATURE_SETS[feature_type])
        for lang in languages
    }
    dist_fn = DISTANCE_FNS[distance_metric]
    D = np.zeros((n, n))
    for i, li in enumerate(languages):
        for j, lj in enumerate(languages):
            if i != j:
                D[i, j] = dist_fn(feature_vecs[li], feature_vecs[lj])
    return D


def compute_euclidean_distances(languages: List[str], feature_type: str) -> np.ndarray:
    """Compute pairwise Euclidean distances for fam/geo features."""
    n = len(languages)
    feature_vecs = {}
    for lang in languages:
        raw = get_language_features(lang, FEATURE_SETS[feature_type])
        feature_vecs[lang] = np.array(
            [float(x) for x in raw if x != "--"], dtype=float
        )
    D = np.zeros((n, n))
    for i, li in enumerate(languages):
        for j, lj in enumerate(languages):
            if i != j:
                vi, vj = feature_vecs[li], feature_vecs[lj]
                min_len = min(len(vi), len(vj))
                if min_len > 0:
                    D[i, j] = np.linalg.norm(vi[:min_len] - vj[:min_len])
    return D


def compute_all_distances(languages: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute all 14 linguistic distance matrices for the given languages.

    Returns a dict mapping distance_name -> (n_lang x n_lang) ndarray.
    Distance names: 'hamming-syntax', ..., 'anderberg-inventory', 'fam', 'geo'.
    """
    distances: Dict[str, np.ndarray] = {}
    for feat in TYPOLOGICAL_FEATURES:
        for metric in BINARY_METRICS:
            name = f"{metric}-{feat}"
            distances[name] = compute_pairwise_distances(languages, feat, metric)
    for feat in ["fam", "geo"]:
        distances[feat] = compute_euclidean_distances(languages, feat)
    return distances


def combined_metric(distances: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute the combined distance metric from the paper:
        d_comb = 0.4 * d_ander-syntax + 0.2 * d_inner-phonology + 0.4 * d_ander-inventory

    This metric achieves the highest correlation (>0.6) across all task/scale settings.
    """
    return (
        0.4 * distances["anderberg-syntax"]
        + 0.2 * distances["inner-phonology"]
        + 0.4 * distances["anderberg-inventory"]
    )


def pearson_correlation(dist_matrix: np.ndarray, transfer_matrix: np.ndarray) -> float:
    """Compute Pearson correlation between non-diagonal entries."""
    n = dist_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    d_vec = dist_matrix[mask]
    t_vec = transfer_matrix[mask]
    if np.std(d_vec) < 1e-10 or np.std(t_vec) < 1e-10:
        return 0.0
    corr, _ = pearsonr(d_vec, t_vec)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_all_correlations(
    distances: Dict[str, np.ndarray],
    transfer_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute Pearson correlation of each distance metric with transfer scores."""
    return {name: pearson_correlation(D, transfer_scores) for name, D in distances.items()}


def optimize_combined_weights(
    distances: Dict[str, np.ndarray],
    transfer_scores: np.ndarray,
    languages: List[str],
) -> Dict[str, float]:
    """
    Find optimal weights for the combined metric via constrained correlation maximization:
        max_{w} corr(sum_i w_i * d_i, transfer_score)
        s.t. 0 <= w_i <= 1, sum w_i = 1

    Uses scipy SLSQP optimizer.
    """
    dist_names = list(distances.keys())
    n_dists = len(dist_names)
    n_langs = len(languages)

    mask = ~np.eye(n_langs, dtype=bool)
    dist_vecs = np.stack([distances[k][mask] for k in dist_names], axis=1)
    score_vec = transfer_scores[mask]

    def neg_corr(w: np.ndarray) -> float:
        combined = dist_vecs @ w
        if np.std(combined) < 1e-10:
            return 0.0
        corr, _ = pearsonr(combined, score_vec)
        return -float(corr) if not np.isnan(corr) else 0.0

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_dists
    w0 = np.ones(n_dists) / n_dists

    result = minimize(neg_corr, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return {dist_names[i]: float(result.x[i]) for i in range(n_dists)}
