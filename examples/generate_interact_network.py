"""
Example of INtERAcT usage.

Generate a network using an embedding and a list of words.
The embedding used is in Word2Vec binary format and has been
trained on a set of prostate cancer abstract.
"""
import os
from interact import get_network_from_embedding_using_interact
from interact.embedding import read_embedding_df
from interact.nn_tree import NearestNeighborsTree


genes = [
    'PTEN', 'AR', 'GSK3B', 'FOXO1',
    'TP53', 'CTNNB1', 'KRAS', 'CDK2'
]
data_path = 'data/'
embedding_filepath = os.path.join(data_path, (
    'kegg_prostate_cancer_abstract-corpus-wnl'
    '-sw-pubtator-bigram_w_9_size_500_c_50.bin'
))
# embedding
embedding_df = read_embedding_df(embedding_filepath)
# make sure gene names are capitalized
embedding_df.index = [
    index.upper() for index in embedding_df.index
]
# nn
nn_tree = NearestNeighborsTree(embedding_df)
# retrieve clusters from the tree
number_of_clusters = len(nn_tree.cluster_series.unique())
# query
interaction_table = (
    get_network_from_embedding_using_interact(
        genes, embedding_df, nn_tree,
        number_of_clusters=number_of_clusters,
        number_of_neighbors=2000
    )
)
print(interaction_table.sort_values(by='intensity'))
