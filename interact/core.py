"""Methods used for INtERAcT."""
import numpy as np
import pandas as pd
from collections import Counter
from numpy.linalg import norm
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from .nn_tree import NeighborsMode


def _nn_data_to_clusters_counts(
    nn_data, number_of_clusters, number_of_neighbors
):
    clusters_counts = np.zeros(number_of_clusters)
    counts = pd.Series(Counter(nn_data))
    clusters_counts[counts.index] = counts.values
    return clusters_counts


def jensen_shannon_divergence(p, q, normalize=True):
    """Compute Jensen-Shannon divergence given two pmfs."""
    if normalize:
        p = p / norm(p, ord=1)
        q = q / norm(q, ord=1)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def _compute_divergence_matrix(words_clusters_distributions, n=None):
    if n is None:
        n = words_clusters_distributions.shape[0]
    divergence_matrix = np.zeros((n, n))
    divergence = .0
    for i in range(n):
        for j in range(i):
            divergence = jensen_shannon_divergence(
                words_clusters_distributions.values[i],
                words_clusters_distributions.values[j]
            )
            divergence_matrix[i, j] = divergence
            divergence_matrix[j, i] = divergence
    return divergence_matrix


def _divergence_matrix_to_table(
    divergence_matrix, proteins_list, n=None, interaction_symbol='<->'
):
    if n is None:
        n = divergence_matrix.shape[0]
    return pd.DataFrame(
        [
            [
                interaction_symbol.join(sorted(
                    [proteins_list[i], proteins_list[j]]
                )), divergence_matrix[j][i]
            ]
            for i in range(n)
            for j in range(i)
        ], columns=['interaction', 'divergence']
    ).set_index('interaction')


def _divergence_table_to_interaction_df(
    divergence_table, interaction_symbol='<->',
    alpha=7.5, beta=0.0
):
    e1s = {}
    e2s = {}
    intensity = {}
    divergence_table['intensity'] = (
        np.exp(-alpha*divergence_table['divergence'] + beta)
    )
    for index, value in zip(
        divergence_table.index, divergence_table['intensity']
    ):
        interaction_table_index = index.upper()
        e1, e2 = interaction_table_index.split(interaction_symbol)
        e1s[interaction_table_index] = e1
        e2s[interaction_table_index] = e2
        intensity[interaction_table_index] = value
    return pd.DataFrame({
        "e1": e1s,
        "e2": e2s,
        "intensity": intensity
    })


def _distance_matrix_to_table(
    distance_matrix, proteins_list, n=None, interaction_symbol='<->'
):
    if n is None:
        n = distance_matrix.shape[0]
    return pd.DataFrame(
        [
            [
                interaction_symbol.join(sorted(
                    [proteins_list[i], proteins_list[j]]
                )), distance_matrix[j][i]
            ]
            for i in range(n)
            for j in range(i)
        ], columns=['interaction', 'distance']
    ).set_index('interaction')


def _distance_table_to_interaction_df(
    distance_table, interaction_symbol='<->'
):
    e1s = {}
    e2s = {}
    intensity = {}
    distance_table['intensity'] = 1 / distance_table['distance']
    minimum, maximum = (
        min(distance_table['intensity']), max(distance_table['intensity'])
    )
    distance_table['intensity'] = (
        distance_table['intensity'] - minimum
    ) / (maximum - minimum)
    for index, value in zip(distance_table.index, distance_table['intensity']):
        interaction_table_index = index.upper()
        e1, e2 = interaction_table_index.split(interaction_symbol)
        e1s[interaction_table_index] = e1
        e2s[interaction_table_index] = e2
        intensity[interaction_table_index] = value
    return pd.DataFrame({
        "e1": e1s,
        "e2": e2s,
        "intensity": intensity
    })


def _interaction_df_to_edge_weight_list(interaction_table, threshold=0.0):
    """Convert from df to edge_list."""
    edge_weight_list = [
        tuple(sorted([row['e1'], row['e2']]) + [row['intensity']])
        for idx, row in interaction_table.iterrows()
        if row['intensity'] > threshold
    ]
    return edge_weight_list


def get_network_from_embedding_using_interact(
    word_list, embedding_df, nn_tree,
    number_of_clusters, number_of_neighbors=2000
):
    """Return interaction dataframe using INtERAcT."""
    vectors = embedding_df.loc[word_list]
    nn_data_list = nn_tree.kneighbors(
        X=vectors.values,
        k=number_of_neighbors,
        mode=NeighborsMode.CLUSTERS
    )
    words_clusters_distributions = pd.DataFrame(
        np.array([
            _nn_data_to_clusters_counts(
                nn_data, number_of_clusters,
                number_of_neighbors
            )
            for nn_data in nn_data_list
        ]) / number_of_neighbors,
        index=word_list,
        columns=[
            'C{}'.format(i)
            for i in range(number_of_clusters)
        ]
    )
    divergence_matrix = _compute_divergence_matrix(
        words_clusters_distributions
    )
    divergence_table = _divergence_matrix_to_table(
        divergence_matrix, word_list
    ).sort_values(by='divergence')
    return _divergence_table_to_interaction_df(divergence_table)


def get_network_from_embedding_using_distance_metric(
    word_list, embedding_df, metric='euclidean'
):
    """Return interaction dataframe using a distance."""
    vectors = embedding_df.loc[word_list]
    distance = squareform(pdist(vectors.values, metric=metric))
    distance_table = _distance_matrix_to_table(
        distance, word_list
    ).sort_values(by='distance')
    interaction_df = _distance_table_to_interaction_df(
        distance_table
    ).sort_values(by='intensity')
    return interaction_df
