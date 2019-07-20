"""I/O utilities."""
import logging
import numpy as np
import pandas as pd
from enum import Enum
from fasttext import load_model as load_fasttext


logger = logging.getLogger(__name__)


class EmbeddingFormat(Enum):
    """Enum to describe embedding file format."""

    WORD2VEC = 1
    FASTTEXT = 2


def read_embedding_df_word2vec_format(filepath, binary=True, normalize=True):
    """Read embedding from word2vec format."""
    if binary:
        def _parse_lines(fp, number_of_words, dimension):
            bytes_vector_length = np.dtype(np.float32).itemsize * dimension
            for _ in range(number_of_words):
                word = b''
                while True:
                    ch = fp.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise IOError('cannot read word')
                    if ch != b'\n':
                        word += ch
                yield (
                    word.decode(),
                    np.frombuffer(
                        fp.read(bytes_vector_length), dtype=np.float32
                    )
                )
    else:
        def _parse_lines(fp, number_of_words=None, dimension=None):
            for line in fp.readlines():
                splitted_line = line.split()
                yield (
                    splitted_line[0],
                    np.array(map(np.float32, splitted_line[1:]))
                )
    with open(filepath, 'rb' if binary else 'r') as fp:
        number_of_words, dimension = [
            int(parsed)
            for parsed in
            fp.readline().strip().split()
        ]
        words, vectors = zip(*_parse_lines(fp, number_of_words, dimension))
        return pd.DataFrame(
            np.array(vectors),
            index=words
        )


def read_embedding_df_fasttext_format(filepath):
    """Read embedding from fasttext format."""
    model = load_fasttext(filepath)
    return pd.DataFrame({
        word: model.get_word_vector(word)
        for word in model.get_words()
    }).T


def read_embedding_df(
    embedding_filepath, mode=EmbeddingFormat.WORD2VEC,
    normalize=True, **kwargs
):
    """Read embedding pandas DataFrame from filepath."""
    if not isinstance(mode, EmbeddingFormat):
            raise IOError(
                'mode as to be a value from enum EmbeddingFormat'
            )
    if mode == EmbeddingFormat.WORD2VEC:
        embedding_df = read_embedding_df_word2vec_format(
            embedding_filepath, **kwargs
        )
    elif mode == EmbeddingFormat.FASTTEXT:
        embedding_df = read_embedding_df_fasttext_format(
            embedding_filepath
        )
    else:
        raise IOError('EmbeddingFormat mode is not supported.')
    if normalize:
        embedding_df = pd.DataFrame(
            (
                embedding_df.values.T / np.linalg.norm(
                    embedding_df.values, axis=1
                )
            ).T,
            index=embedding_df.index
        )
    return embedding_df
