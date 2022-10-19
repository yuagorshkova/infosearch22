import os
import numpy as np
from setup_logger import logger
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from utils import save_pickle
from utils import load_pickle
from search_class import Search


def load_bm25(bm25_path, data_path, n_docs):
    if not os.path.exists(bm25_path):
        logger.info("Existing BM25 inverted index not found, creating a new one")
        bm25 = BM25(data_path, n_docs)
        save_pickle(bm25, bm25_path)
        logger.info("Finished creating BM25 search engine")
    else:
        logger.info("Using existing BM25 inverted index")
        bm25 = load_pickle(bm25_path)
    return bm25


class BM25(Search):

    def __init__(self, data_path: str, n_docs: int):
        super().__init__(data_path, n_docs)
        self.count_vectorizer = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()

        self.create_inverted_index()

    def create_inverted_index(self):
        """
        Create a BM-25 vectorized document-term matrix.
        """
        if not self.corpus:
            logger.error("Create corpus first!")
        logger.info("Creating the term-document matrix using BM25 method")

        tf = self.count_vectorizer.fit_transform(self.corpus)
        self.tfidf_vectorizer.fit(self.corpus)
        idf = self.tfidf_vectorizer.idf_
        self.vocab = self.tfidf_vectorizer.vocabulary_

        len_d = tf.sum(axis=1)
        avg_dl = len_d.mean()

        k = 2
        b = 0.75

        vector_of_k = np.full(idf.shape, k + 1)
        matrix = tf.multiply(idf).multiply(vector_of_k).tocsr()

        del_element = (k * (1 - b + b * len_d / avg_dl))
        for i, j in zip(*matrix.nonzero()):
            matrix[i, j] /= (tf[i, j] + del_element[i, 0])
        self.matrix = matrix

    def get_query_vectors(self, queries: List[str]):
        """
        :param queries:
        :return: BoW representations of the queries in csc
        """
        preprocessed_queries = [self.preprocessor.preprocess_text(query) for query in queries]
        print(preprocessed_queries)
        return self.count_vectorizer.transform(preprocessed_queries).transpose()

    def get_closeness(self, query_vectors: np.ndarray):
        """
        :param query_vectors:
        :return: array of closeness between query and each document
        """
        return (self.matrix * query_vectors).toarray()
