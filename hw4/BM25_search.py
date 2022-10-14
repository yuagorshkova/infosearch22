import numpy as np
from BM25_inverted_index import InvertedIndex25


class Search25:

    def __init__(self, inverted_index: InvertedIndex25):
        self.inverted_index = inverted_index

    def get_query_vector(self, queries: list):
        """
        :param query:
        :return: BoW representation of the query in csc
        """
        preprocessed_queries = [self.inverted_index.preprocessor.preprocess_text(query) for query in queries]
        return self.inverted_index.count_vectorizer.transform(preprocessed_queries).transpose()

    def get_bm25_distance(self, query_vector: np.ndarray):
        """
        :param query_vector:
        :return: array of BM25 values computed between query and each document
        """
        return (self.inverted_index.matrix * query_vector).toarray()

    def find_closest_docs(self, query: str, n: int = None):
        """
        :param query:
        :param n: number of result docs to list
        :return: n most similar docs
        """
        if not n:
            n = self.inverted_index.matrix.shape[0]

        query_vector = self.get_query_vector([query])
        distances = self.get_bm25_distance(query_vector).ravel()
        sorted_indices = np.argsort(distances)[:-n-1:-1]
        return self.inverted_index.filename_index[sorted_indices]
