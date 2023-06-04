"""
Language clustering and selection for ZSCL-M (multi-transfer).

Uses k-medoids clustering on the combined linguistic distance matrix (d_comb)
to partition languages into clusters and identify medoid languages.

MINION: 2 clusters — tur* (Korean/Hindi/Japanese/Turkish) and por* (English/Polish/Spanish/Portuguese/Swedish)
SMiLER: 4 clusters — ita*, nld*, fas*, medoids* (inter-cluster)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def kmedoids_cluster(
    dist_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 300,
) -> Tuple[np.ndarray, List[int]]:
    """
    K-medoids clustering using a precomputed distance matrix.

    Args:
        dist_matrix: (n, n) pairwise distance matrix
        n_clusters: number of clusters
        random_state: random seed
        max_iter: maximum iterations

    Returns:
        (labels, medoid_indices): cluster labels and indices of medoids
    """
    try:
        from sklearn_extra.cluster import KMedoids
        km = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            random_state=random_state,
            max_iter=max_iter,
        )
        km.fit(dist_matrix)
        return km.labels_, km.medoid_indices_.tolist()
    except ImportError:
        return _kmedoids_custom(dist_matrix, n_clusters, random_state, max_iter)


def _kmedoids_custom(
    dist_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 300,
) -> Tuple[np.ndarray, List[int]]:
    """Fallback k-medoids implementation (PAM algorithm)."""
    rng = np.random.RandomState(random_state)
    n = dist_matrix.shape[0]

    medoid_indices = rng.choice(n, n_clusters, replace=False).tolist()

    for _ in range(max_iter):
        labels = _assign_clusters(dist_matrix, medoid_indices)
        new_medoids = []
        for k in range(n_clusters):
            cluster_mask = labels == k
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                new_medoids.append(medoid_indices[k])
                continue
            intra_dists = dist_matrix[np.ix_(cluster_indices, cluster_indices)].sum(axis=1)
            best_local = np.argmin(intra_dists)
            new_medoids.append(cluster_indices[best_local])
        if new_medoids == medoid_indices:
            break
        medoid_indices = new_medoids

    labels = _assign_clusters(dist_matrix, medoid_indices)
    return labels, medoid_indices


def _assign_clusters(dist_matrix: np.ndarray, medoid_indices: List[int]) -> np.ndarray:
    """Assign each point to its closest medoid."""
    dists_to_medoids = dist_matrix[:, medoid_indices]
    return np.argmin(dists_to_medoids, axis=1)


def build_language_graph(
    languages: List[str],
    cluster_labels: np.ndarray,
    medoid_indices: List[int],
    connect_medoids: bool = True,
) -> np.ndarray:
    """
    Build binary adjacency matrix for the language-relational graph.

    Intra-cluster edges: all languages within the same cluster are connected.
    Inter-cluster edges: medoid languages are connected to each other (red edges in paper,
    added to facilitate inter-cluster transfer).

    Args:
        languages: list of language codes
        cluster_labels: cluster assignment per language
        medoid_indices: index of medoid language per cluster
        connect_medoids: whether to add edges between medoids (for ZSCL-R)

    Returns:
        A: (n_lang, n_lang) binary adjacency matrix
    """
    n = len(languages)
    A = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j and cluster_labels[i] == cluster_labels[j]:
                A[i, j] = 1.0

    if connect_medoids:
        for i in medoid_indices:
            for j in medoid_indices:
                if i != j:
                    A[i, j] = 1.0

    return A


def select_source_languages(
    languages: List[str],
    cluster_labels: np.ndarray,
    medoid_indices: List[int],
    mode: str = "inter",
    target_cluster: Optional[int] = None,
    n_sources: Optional[int] = None,
    random_state: int = 42,
) -> List[str]:
    """
    Select source languages for ZSCL-M.

    Modes:
        'inter': medoids of all clusters (best for inter-cluster transfer)
        'intra': languages from within the target cluster (best for intra-cluster transfer)
        'random': random baseline (n_sources randomly chosen languages)
    """
    rng = np.random.RandomState(random_state)

    if mode == "inter":
        return [languages[i] for i in medoid_indices]

    elif mode == "intra":
        if target_cluster is None:
            raise ValueError("target_cluster required for 'intra' mode")
        cluster_members = [i for i, c in enumerate(cluster_labels) if c == target_cluster]
        n = n_sources if n_sources is not None else len(cluster_members)
        chosen = rng.choice(cluster_members, size=min(n, len(cluster_members)), replace=False)
        return [languages[i] for i in chosen]

    elif mode == "random":
        n = n_sources if n_sources is not None else len(medoid_indices)
        chosen = rng.choice(len(languages), size=n, replace=False)
        return [languages[i] for i in chosen]

    else:
        raise ValueError(f"Unknown language selection mode: '{mode}'")


def get_cluster_summary(
    languages: List[str],
    cluster_labels: np.ndarray,
    medoid_indices: List[int],
) -> Dict:
    """Return a human-readable summary of the clustering results."""
    n_clusters = len(medoid_indices)
    summary = {}
    for k in range(n_clusters):
        medoid_lang = languages[medoid_indices[k]]
        members = [languages[i] for i, c in enumerate(cluster_labels) if c == k]
        summary[f"cluster_{k}"] = {
            "medoid": medoid_lang,
            "members": members,
            "size": len(members),
        }
    return summary
