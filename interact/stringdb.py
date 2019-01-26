"""Methods related to string-db."""
import numpy as np
import gzip
import os
from .generic import download_from_url, filter_interaction_table_by_labels


STRING_PROTEIN_LINKS_URL = (
    'https://string-db.org/download/protein.links.full.v10.5/' +
    '9606.protein.links.full.v10.5.txt.gz'
)

STRING_PROTEIN_ACTIONS_URL = (
    'https://string-db.org/download/protein.actions.v10.5/' +
    '9606.protein.actions.v10.5.txt.gz'
)


def interaction_table_from_string(
    gene_list, string_interactions, threshold=0.0,
    use_as_true_network=False, interaction_symbol='<->'
):
    """Return string interactions given a list of genes."""
    interactions = filter_interaction_table_by_labels(
        string_interactions, gene_list
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
    interactions['intensity'] /= 1000
    if threshold > 0.0:
        interactions = interactions[interactions['intensity'] > threshold]
    if use_as_true_network:
        interactions['intensity'] = 1.0
    return interactions[['e1', 'e2', 'intensity']]  # set order of columns


def list_stringdb_interaction_sources(target_path=None):
    """List types of interaction sources available."""
    local_filepath = download_from_url(STRING_PROTEIN_LINKS_URL, target_path)
    with gzip.open(local_filepath) as fp:
        print(fp.readline().decode().strip().split(' ')[2:])


def get_stringdb_links(
    entity_dict=None, interaction_source='combined_score',
    min_score=0, return_score=False, target_path=None
):
    """Retrieve string interactions."""
    local_filepath = download_from_url(STRING_PROTEIN_LINKS_URL, target_path)
    # build list of interactions
    interactions = []
    with gzip.open(local_filepath) as fp:
        interaction_sources = fp.readline().decode().strip().split(' ')
        weight_index = interaction_sources.index(interaction_source)
        for line in fp:
            splitted_line = line.strip().decode().split(' ')
            gene1 = splitted_line[0].split('.')[1]
            gene2 = splitted_line[1].split('.')[1]
            weight = int(splitted_line[weight_index])
            if entity_dict:
                gene1 = entity_dict.get(gene1)
                gene2 = entity_dict.get(gene2)
            if gene1 and gene2 and weight > min_score:
                if return_score:
                    interactions.append(
                        (gene1, gene2, weight)
                    )
                else:
                    interactions.append(
                        (gene1, gene2)
                    )
    return interactions


class ActionFilter(object):
    """Can be passed to get_stringdb_actions to filter interactions."""

    def __init__(self, filter_modes=None, filter_pathectional=None):
        """Initialize with filter_modes and filter_pathectional."""
        self.filter_modes = filter_modes
        self.filter_pathectional = filter_pathectional

    def apply_filter(self, mode, is_pathectional):
        """Apply filter."""
        passed = True
        if (
            self.filter_modes and
            mode not in self.filter_modes
        ):
            passed = False
        if (
            self.filter_pathectional and
            is_pathectional == self.filter_pathectional
        ):
            passed = False
        return passed


def list_stringdb_interaction_modes(target_path=None):
    """List types of interaction modes available."""
    local_filepath = download_from_url(STRING_PROTEIN_ACTIONS_URL, target_path)
    with gzip.open(local_filepath) as fp:
        fp.readline()  # skip header
        return {
            line.strip().decode().split('\t')[2] for line in fp
        }


def get_stringdb_actions(
    ensembl_dict=None, action_filter=None, return_mode=False, target_path=None
):
    """Retrieve actions from stringdb."""
    local_filepath = download_from_url(STRING_PROTEIN_ACTIONS_URL, target_path)
    interactions = []
    with gzip.open(local_filepath) as fp:
        fp.readline()  # skip header
        for line in fp:
            splitted_line = line.strip().decode().split('\t')
            gene1 = splitted_line[0].split('.')[1]
            gene2 = splitted_line[1].split('.')[1]
            mode = splitted_line[2]
            is_pathectional = splitted_line[4]
            if ensembl_dict:
                gene1 = ensembl_dict.get(gene1)
                gene2 = ensembl_dict.get(gene2)
            filter_passed = True
            if action_filter is not None:
                filter_passed = action_filter.apply_filter(
                    mode, is_pathectional
                )
            if gene1 and gene2 and filter_passed:
                if return_mode:
                    interactions.append(
                        (gene1, gene2, mode)
                    )
                else:
                    interactions.append(
                        (gene1, gene2)
                    )
    return interactions
