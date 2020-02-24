"""Generate network from keyed vectors."""
import pandas as pd
from gensim.models import KeyedVectors
from interact.uniprot import dict_from_uniprot
from interact import get_network_from_embedding_using_interact
from interact.embedding import read_embedding_df
from interact.nn_tree import NearestNeighborsTree

# parameters
number_of_neighbors = 2000
# read keyed vectors
keyed_vectors = KeyedVectors.load('medulloblastoma/embeddings.kv', mmap='r')
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
interaction_table.sort_values(by='intensity', ascending=False).to_csv('medulloblastoma/interact.csv')

