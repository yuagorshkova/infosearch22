import numpy as np
from inverted_index import InvertedIndex
from setup_model import get_sentences_vectors
from sklearn.metrics.pairwise import cosine_similarity


class Search:

    def __init__(self, inverted_index: InvertedIndex):
        self.inverted_index = inverted_index

    def get_query_vector(self, queries: list):
        """
        :param query:
        :return: BERT representation of the query
        """
        preprocessed_queries = [self.inverted_index.preprocessor.preprocess_text(query) for query in queries]
        return get_sentences_vectors(preprocessed_queries,
                                     self.inverted_index.tokenizer,
                                     self.inverted_index.model)

    def get_closeness(self, query_vectors: np.ndarray):
        """
        :param query_vectors:
        :return: array of cosine similarities between query vectors and the collection docs
        """
        return cosine_similarity(self.inverted_index.matrix, query_vectors)

    def find_closest_docs(self, query: str, n: int = None):
        """
        :param query:
        :param n: number of result docs to list
        :return: n most similar docs
        """
        if not n:
            n = self.inverted_index.matrix.shape[0]

        query_vector = self.get_query_vector([query])
        closeness = self.get_closeness(query_vector).ravel()
        sorted_indices = np.argsort(closeness)[:-n-1:-1]
        return self.inverted_index.filename_index[sorted_indices]
