import numpy as np
from scipy.spatial import distance
from inverted_index import InvertedIndex


class Search:

    def __init__(self, inverted_index: InvertedIndex):
        self.inverted_index = inverted_index

    def get_query_vector(self, query: str):
        """
        :param query:
        :return: TF-iDF representation of the query
        """
        preprocessed_query = self.inverted_index.preprocessor.preprocess_text(query)
        return self.inverted_index.vectorizer.transform([preprocessed_query]).toarray()

    def get_cosine_distance(self, query_vector: np.ndarray):
        """
        :param query_vector:
        :return: array of cosine distances from query vector to the collection docs
        """
        return np.apply_along_axis(distance.cosine, 1, self.inverted_index.matrix.toarray(), query_vector[0])

    def find_closest_docs(self, query: str, n: int = None):
        """
        :param query:
        :param n: number of result docs to list
        :return: n most similar docs
        """
        if not n:
            n = self.inverted_index.matrix.shape[0]

        query_vector = self.get_query_vector(query)
        distances = self.get_cosine_distance(query_vector)
        sorted_indices = np.argsort(distances)[:n]
        return self.inverted_index.filename_index[sorted_indices]
