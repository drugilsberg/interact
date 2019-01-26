"""Methods used to build ROC."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# seaborn settings
sns.set_style("white")
sns.set_context("paper")
color_palette = sns.color_palette("colorblind")
sns.set_palette(color_palette)


def _get_total_undirected_interactions(n):
    return n * (n - 1) / 2


def _check_index(index, labels_set, interaction_symbol='<->'):
    e1, e2 = index.split(interaction_symbol)
    return (e1 in labels_set and e2 in labels_set)


def _filter_indices_with_labels(indexes, labels, interaction_symbol='<->'):
    labels_set = set(labels)
    filtering = pd.Series([
        _check_index(index, labels_set, interaction_symbol)
        for index in indexes
    ])
    return indexes[filtering]


def _is_index_diagonal(index, interaction_indices='<->'):
    a_node, another_node = index.split(interaction_indices)
    return a_node == another_node


def _get_evaluation_on_given_labels(
    labels, true_interactions, predicted_interactions, no_self_loops=True
):
    total_interactions = _get_total_undirected_interactions(len(labels))
    interaction_indices = list(
        set(
            _filter_indices_with_labels(predicted_interactions.index, labels) |
            _filter_indices_with_labels(true_interactions.index, labels)
        )
    )
    if no_self_loops:
        interaction_indices = [
            index
            for index in interaction_indices
            if not _is_index_diagonal(index)
        ]
    predicted_interactions = predicted_interactions.reindex(
        interaction_indices
    ).fillna(0.0)
    true_interactions = true_interactions.reindex(
        interaction_indices
    ).fillna(0.0)
    zero_interactions = int(total_interactions) - len(interaction_indices)
    y = np.append(true_interactions.values, np.zeros((zero_interactions)))
    scores = np.append(
        predicted_interactions.values, np.zeros((zero_interactions))
    )
    return y, scores


def get_roc_df(
    pathway_name, method_name, true_interactions, predicted_interactions,
    number_of_roc_points=100
):
    """Return dataframe that can be used to plot a ROC curve."""
    labels = {
        gene
        for genes in [
            true_interactions.e1, predicted_interactions.e1,
            true_interactions.e2, predicted_interactions.e2
        ]
        for gene in genes
    }
    y, scores = _get_evaluation_on_given_labels(
        labels, true_interactions.intensity,
        predicted_interactions.intensity
    )
    # print(method_name, y, scores)
    reference_xx = np.linspace(0, 1, number_of_roc_points)
    if sum(y) > 0:
        xx, yy, threshold = roc_curve(y, scores)
        print(method_name, y, scores, threshold, xx, yy)
        area_under_curve = auc(xx, yy)
        yy = np.interp(reference_xx, xx, yy)
    else:
        yy = reference_xx
        area_under_curve = 0.5  # worst
    roc_df = pd.DataFrame({
        'pathway': number_of_roc_points * [pathway_name],
        'method': (
            number_of_roc_points * [method_name]
        ),
        'YY': yy,
        'XX': reference_xx.tolist()
    })
    return roc_df, area_under_curve


def plot_roc_curve_from_df(
    df, auc_dict_list=None, output_filepath=None, figsize=(6, 6)
):
    """From a df with multiple methods plot a roc curve using sns.tspot."""
    xlabel = 'False Discovery Rate'
    ylabel = 'True Positive Rate'
    title = 'Receiver Operating Characteristic'

    # rename method name to include AUC to show it in legend
    if auc_dict_list:
        for method in auc_dict_list.keys():
            mean_auc = np.mean(auc_dict_list[method])
            method_indices = df['method'] == method
            df['mean_auc'] = mean_auc
            df.loc[method_indices, 'method'] = (
                '{} '.format(
                    method.capitalize()
                    if method != 'INtERAcT'
                    else method
                ) +
                'AUC=%0.2f' % mean_auc
            )
        df = df.sort_values(by='method')

    df.rename(columns={'method': ''}, inplace=True)  # to avoid legend title
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.tsplot(
        data=df, time='XX', value='YY',
        condition='', unit='pathway', legend=True
    )
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
