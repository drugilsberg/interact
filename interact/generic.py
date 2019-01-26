"""General utilities used in interact."""
import os
import logging
import urllib.request
import csv
import tempfile
import numpy as np
import pandas as pd
import networkx as nx


logger = logging.getLogger(__name__)


def filter_interaction_table_by_labels(interaction_table, labels):
    """Filter interaction table using nodes labels."""
    pattern = r'|'.join([r'^{}$'.format(label) for label in labels])
    return interaction_table[
        interaction_table['e1'].str.match(pattern) &
        interaction_table['e2'].str.match(pattern)
    ]


def interaction_table_from_interactions_df(
    interactions_df, entities_list, threshold=0.0, scaling=1.0,
    force_unweigthed=False, interaction_symbol='<->'
):
    """
    Return interactions given a list of entities.

    The input interaction df is assumed to have columns named as
    follows:
    ```
    e1,e2,intensity
    ```
    """
    interactions = filter_interaction_table_by_labels(
        interactions_df, entities_list
    ).copy()
    # add another column where entity names are sorted
    interactions['e_sorted'] = interactions.apply(
        lambda row: interaction_symbol.join(sorted([row.e1, row.e2])), axis=1
    )
    interactions = interactions.groupby(
        by=['e_sorted']
    ).apply(np.mean)
    # readd single entities
    e1s, e2s = list(zip(*map(
        lambda index: index.split(interaction_symbol),
        list(interactions.index)
    )))
    interactions['e1'] = e1s
    interactions['e2'] = e2s
    interactions.index.name = None
    interactions['intensity'] *= scaling
    if threshold > 0.0:
        interactions = interactions[interactions['intensity'] > threshold]
    if force_unweigthed:
        interactions['intensity'] = 1.0
    return interactions[['e1', 'e2', 'intensity']]  # set order of columns


def read_gmt(gmt_filepath):
    """Read gsea file with gene sets."""
    gmt_dict = {}
    with open(gmt_filepath) as fp:
        for line in fp:
            splitted = line.strip().split('\t')
            gmt_dict[splitted[0]] = splitted[2:]
    return gmt_dict


def read_interactions(data_path, pathway, method):
    """Read interactions from csv file."""
    return pd.read_csv(os.path.join(
        data_path, '{}_{}.csv'.format(pathway, method)
    ), index_col=0)


def download_from_url(
    url, target_path=None, target_filename=None, overwrite=False
):
    """
    Download files to target_path.

    Input is a dictionary with names and urls. Output is a dictionary with
    the same key but with the path to the downloaded file.
    """
    if target_path is None:
        target_path = tempfile.mkdtemp()
        logger.info('setting target_path to download {}'.format(
            target_path
        ))
    elif not os.path.exists(target_path):
        raise IOError(
            'target_path {} '.format(target_path) +
            ' does not exist. Specify valid directory.'
        )
    if target_filename is None:
        target_filename = os.path.basename(url)
    target_filepath = os.path.join(target_path, target_filename)
    if not os.path.exists(target_filepath) or overwrite is True:
        logger.info('attempt to download {} to {}'.format(
            url, target_filepath)
        )
        # needed to add header because string refused download otherwise
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        try:
            urllib.request.urlretrieve(url, target_filepath)
        except Exception as exc:
            logger.exception('download failed.')
    return target_filepath


def write_list_of_tuples_to_csv(data, filepath):
    """Write list of tuples to csv file."""
    with open(filepath, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerows(data)


def interactions_to_tex(filename, interactions, top_n=50):
    """Dump interactions into a tex table."""
    with open(os.path.join(
        filename
    ), 'w') as fp:
        for index, row in interactions[:top_n].iterrows():
            fp.write(
                '{} & {} & {:.2f} \\\ \hline \n'.format(
                    row.e1, row.e2, row.intensity
                )
            )


def interactions_to_edges(interactions, interaction_symbol='<->'):
    """Split interaction df into tuples of edges."""
    return {
        tuple(interaction.split(interaction_symbol))
        for interaction in interactions.index
    }


def interactions_to_graph(interactions):
    """Get Networkx graph given interactions."""
    G = nx.Graph()
    G.add_weighted_edges_from(interactions.to_records(index=False))
    return G
