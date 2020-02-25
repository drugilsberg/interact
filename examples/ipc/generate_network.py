"""Generate network from keyed vectors."""
import os
import argparse
import pandas as pd
from gensim.models import KeyedVectors
from interact.uniprot import dict_from_uniprot
from interact import get_network_from_embedding_using_interact
from interact.embedding import read_embedding_df
from interact.nn_tree import NearestNeighborsTree

# parse command line arguments
arg_parser = argparse.ArgumentParser(
    description='Run interact on KeyedVectors from gensim.'
)
arg_parser.add_argument(
    '-i', '--input_filepath', required=True,
    help='path to the data.'
)
arg_parser.add_argument(
    '-o', '--output_filename', required=True,
    help='name of the output file.'
)
arg_parser.add_argument(
    '-n', '--number_of_neighbors',
    required=False, default=2000, type=int,
    help='standardize the data. Defaults to 2000.'
)
args = arg_parser.parse_args()
# adjust arguments
number_of_neighbors = args.number_of_neighbors
input_filepath = args.input_filepath
output_filename = args.output_filename
# construct output filepath
output_filepath = os.path.join(os.path.dirname(input_filepath), output_filename)
# read keyed vectors
keyed_vectors = KeyedVectors.load(input_filepath, mmap='r')
# retrieve gene names from UniProt
name_mapping = dict_from_uniprot()
gene_names_set = set(map(str.lower, name_mapping.values()))
# filter words
words = [
    word
    for word in keyed_vectors.index2word
    if word in gene_names_set
]
# index mapping
word2index = {word: index for index, word in enumerate(keyed_vectors.index2word)}
indexes = [word2index[word] for word in words]
selected_words = [str.upper(keyed_vectors.index2word[index]) for index in indexes]
# create embedding dataframe
embedding_df = pd.DataFrame(
    keyed_vectors.vectors,
    index=list(map(str.upper, keyed_vectors.index2word))
)
# nn
nn_tree = NearestNeighborsTree(embedding_df)
# retrieve clusters from the tree
number_of_clusters = len(nn_tree.cluster_series.unique())
# query
interaction_table = (
    get_network_from_embedding_using_interact(
        selected_words, embedding_df, nn_tree,
        number_of_clusters=number_of_clusters,
        number_of_neighbors=number_of_neighbors
    )
)
interaction_table.sort_values(by='intensity', ascending=False).to_csv(output_filepath)

