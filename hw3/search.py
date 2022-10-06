import numpy as np
from inverted_index import InvertedIndex


class Search:

    def __init__(self, inverted_index: InvertedIndex):
        self.inverted_index = inverted_index

    def get_query_vector(self, query: str):
        """
        :param query:
        :return: BoW representation of the query in csc
        """
        preprocessed_query = self.inverted_index.preprocessor.preprocess_text(query)
        return self.inverted_index.count_vectorizer.transform([preprocessed_query]).transpose()

    def get_bm25_distance(self, query_vector: np.ndarray):
        """
        :param query_vector:
        :return: array of BM25 values computed between query and each document
        """

        return self.inverted_index.matrix * query_vector

    def find_closest_docs(self, query: str, n: int = None):
        """
        :param query:
        :param n: number of result docs to list
        :return: n most similar docs
        """
        if not n:
            n = self.inverted_index.matrix.shape[0]

        query_vector = self.get_query_vector(query)
        print(query, query_vector.shape)
        distances = self.get_BM25_distance(query_vector)
        sorted_indices = np.argsort(distances)[::-1][:n]
        return self.inverted_index.filename_index[sorted_indices]
