"""Methods for graph plotting."""
import networkx as nx
import seaborn as sns
import json
from matplotlib import colors
import matplotlib.pyplot as plt
from .generic import interactions_to_edges

# seaborn settings
sns.set_style("white")
sns.set_context("poster")
color_palette = sns.color_palette("colorblind")
sns.set_palette(color_palette)

# colors
blue = colors.rgb2hex(color_palette[0])
green = colors.rgb2hex(color_palette[1])
red = colors.rgb2hex(color_palette[2])
orange = colors.rgb2hex(color_palette[5])

# plot an overlay of two networks
layout_factory = {
    'circular_layout': nx.circular_layout,
    'spring_layout': nx.spring_layout,
    'shell_layout': nx.shell_layout,
    'spectral_layout': nx.spectral_layout,
    'fruchterman_reingold_layout': nx.fruchterman_reingold_layout,
}


def dump_pos(pos, filepath):
    """Dump pos dictionary to json. Convert np array to list first."""
    with open(filepath, 'w') as fp:
        json.dump({
            node: list(coordinate)
            for node, coordinate in pos.items()
        }, fp, indent=2)


def load_pos(filepath):
    """Load pos dictionary from json."""
    with open(filepath, 'r') as fp:
        return json.load(fp)


def plot_graph(
    interactions, pos=None, node_color='orange',
    edge_color='black', title='',
    fixed_weight=None, weight_offset=1, weight_factor=4,
    layout='spring_layout', layout_parameters={},
    figsize=(12, 12), output_filepath=''
):
    """Plot graph from list of pandas df with interactions."""
    G = nx.Graph()
    G.add_weighted_edges_from(interactions.to_records(index=False))
    weights = [
        (
            fixed_weight if fixed_weight is not None else
            (weight_offset + G[u][v]['weight']) * weight_factor
        )
        for u, v in G.edges()
    ]
    if pos is None:
        pos = layout_factory[layout](G, **layout_parameters)
    plt.figure(figsize=figsize)
    plt.title(title)
    nx.draw(
        G, pos=pos, node_color=node_color, width=weights, edge_color=edge_color
    )
    nx.draw_networkx_labels(
        G, pos=pos,
        bbox=dict(
            boxstyle='round', ec=(0.0, 0.0, 0.0),
            alpha=0.9, fc=node_color, lw=1.5
        )
    )
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    return pos


def plot_graph_overlayed(
    true_network_interactions, predicted_network_interactions,
    pos=None, true_edge_color='black',
    tp_edge_color='blue', fp_edge_color='grey',
    node_color='orange', layout='neato', title='',
    interaction_symbol='<->', weight_offset=10, weight_factor=10,
    output_filepath=None, figsize=(12, 12)
):
    """Plot overlayed graph to show TP and FP edges."""
    true_edges = interactions_to_edges(true_network_interactions)
    predicted_edges = interactions_to_edges(predicted_network_interactions)

    # edges sets
    all_edges = true_edges | predicted_edges
    common_edges = true_edges & predicted_edges
    common_edges_indices = [
        '{}{}{}'.format(e1, interaction_symbol, e2)
        for e1, e2 in common_edges
    ]
    true_only_edges = true_edges - common_edges
    predicted_only_edges = predicted_edges - common_edges
    predicted_only_edges_indices = [
        '{}{}{}'.format(e1, interaction_symbol, e2)
        for e1, e2 in predicted_only_edges
    ]
    print('All edges: {}'.format(len(all_edges)))
    print('True edges: {}'.format(len(true_edges)))
    print('Predicted edges: {}'.format(len(predicted_edges)))
    print('Common edges: {}'.format(len(common_edges)))
    print('True-only edges: {}'.format(len(true_only_edges)))
    print('Predicted-only edges: {}'.format(len(predicted_only_edges)))

    # plot nodes
    G = nx.from_edgelist(all_edges)
    if pos is None:
        pos = layout_factory.get(layout, layout_factory['neato'])(G)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos=pos, alpha=0)
    nx.draw_networkx_labels(
        #  G, pos=pos, font_size=18, font_family='arial', font_color='black',
        G, pos=pos,
        bbox=dict(
            boxstyle='round', ec=(0.0, 0.0, 0.0),
            alpha=0.9, fc=node_color, lw=1.5
        )
    )
    widths = list(
        0 + true_network_interactions['intensity'] * 1
    )
    nx.draw_networkx_edges(
        G, pos=pos, edgelist=true_edges, edge_color=true_edge_color,
        alpha=1, width=widths, style='solid'
    )

    # draw TP edges (common edges)
    widths = list(
        weight_offset + predicted_network_interactions.loc[
            common_edges_indices, 'intensity'
        ] * weight_factor
    )
    nx.draw_networkx_edges(
        G, pos=pos, edgelist=common_edges, edge_color=tp_edge_color,
        alpha=0.3, width=widths, style='solid'
    )

    # draw FP edges (predicted_only edged)
    widths = list(
        weight_offset + predicted_network_interactions.loc[
            predicted_only_edges_indices, 'intensity'
        ] * weight_factor
    )
    nx.draw_networkx_edges(
        G, pos=pos, edgelist=predicted_only_edges, edge_color=fp_edge_color,
        alpha=0.3, width=widths, style='dashed'
    )

    # formatting
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    # save
    if output_filepath:
        plt.savefig(output_filepath)
