import numpy as np
from inverted_index import InvertedIndex
from setup_model import get_sentences_vectors
from sklearn.metrics.pairwise import cosine_similarity


class Search:

    def __init__(self, inverted_index: InvertedIndex):
        self.inverted_index = inverted_index

    def get_query_vector(self, query: str):
        """
        :param query:
        :return: BERT representation of the query
        """
        preprocessed_query = self.inverted_index.preprocessor.preprocess_text(query)
        return get_sentences_vectors([preprocessed_query],
                                     self.inverted_index.tokenizer,
                                     self.inverted_index.model)

    def get_cosine_similarity(self, query_vector: np.ndarray):
        """
        :param query_vector:
        :return: array of cosine distances from query vector to the collection docs
        """
        return cosine_similarity(self.inverted_index.matrix, query_vector).ravel()

    def find_closest_docs(self, query: str, n: int = None):
        """
        :param query:
        :param n: number of result docs to list
        :return: n most similar docs
        """
        if not n:
            n = self.inverted_index.matrix.shape[0]

        query_vector = self.get_query_vector(query)
        similarities = self.get_cosine_similarity(query_vector)
        sorted_indices = np.argsort(similarities)[:-n-1:-1]
        return self.inverted_index.filename_index[sorted_indices]
