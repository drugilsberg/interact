"""Search trees related utils."""
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def _map_indices_with_series(indices, series):
    return series[indices].values


def build_tree(vectors, algorithm='kd_tree', metric='minkowski', **kwargs):
    """Build NearestNeighbors tree."""
    kwargs.pop('algorithm', None)
    kwargs.pop('metric', None)
    return NearestNeighbors(
        algorithm=algorithm, metric=metric, **kwargs
    ).fit(vectors)


def cluster_vectors(vectors, k=500, n_init=100, **kwargs):
    """Build NearestNeighbors tree."""
    kwargs.pop('n_clusters', None)
    kwargs.pop('init', None)
    kwargs.pop('n_init', None)
    return KMeans(
        n_clusters=k, init='k-means++', n_init=n_init,
        **kwargs
    ).fit(vectors)


class NeighborsMode(Enum):
    """Enum for nearest neighbours return mode."""
    WORDS = 1
    CLUSTERS = 2
    BOTH = 3


class NearestNeighborsTree(object):
    """Nearest neighbor tree support."""

    tree = None
    word_series = None
    cluster_series = None

    def __init__(self, embedding, algorithm='kd_tree', metric='minkowski',
                 k=500, n_init=10, n_jobs=-1):
        """Build from embedding pd.DataFrame (index: words)."""
        self.word_series = pd.Series(dict(enumerate(embedding.index.values)))
        self.cluster_series = pd.Series(dict(enumerate(
            cluster_vectors(
                embedding.values, k=k, n_init=n_init, n_jobs=n_jobs
            ).labels_
        )))
        self.tree = build_tree(
            embedding.values, algorithm=algorithm,
            metric=metric, n_jobs=n_jobs
        )

    def kneighbors(self, X=None, k=5, mode=NeighborsMode.BOTH):
        """Get k neighbors from query points."""
        if not isinstance(mode, NeighborsMode):
            raise RuntimeError(
                'mode as to be a value from enum NeighborsMode'
            )
        neighbors_indices = self.tree.kneighbors(
            X=X, n_neighbors=k, return_distance=False
        )

        if mode == NeighborsMode.WORDS:
            return np.array([
                _map_indices_with_series(indices, self.word_series)
                for indices in neighbors_indices
            ])
        elif mode == NeighborsMode.CLUSTERS:
            return np.array([
                _map_indices_with_series(indices, self.cluster_series)
                for indices in neighbors_indices
            ])
        elif mode == NeighborsMode.BOTH:
            return (
                np.array([
                    _map_indices_with_series(indices, self.word_series)
                    for indices in neighbors_indices
                ]),
                np.array([
                    _map_indices_with_series(indices, self.cluster_series)
                    for indices in neighbors_indices
                ])
            )
        else:
            raise RuntimeError('invalid return mode')
