import os
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from setup_logger import logger
from utils import save_pickle
from utils import load_pickle
from search_class import Search


def load_tfidf(tfidf_path, data_path, n_docs):
    if not os.path.exists(tfidf_path):
        logger.info("Existing TF-iDFsearch engine not found, creating a new one")
        tfidf = TFiDF(data_path, n_docs)
        save_pickle(tfidf, tfidf_path)
        logger.info("Finished creating TF-iDF search engine")
    else:
        logger.info("Using existing TF-iDF inverted index")
        tfidf = load_pickle(tfidf_path)
    return tfidf


class TFiDF(Search):

    def __init__(self, data_path: str, n_docs: int):
        super().__init__(data_path, n_docs)
        self.tfidf_vectorizer = TfidfVectorizer()

        self.create_inverted_index()

    def create_inverted_index(self):
        """
        Create a tf-idf vectorized document-term matrix.
        """
        if not self.corpus:
            logger.error("Create corpus first!")
        logger.info("Creating the term-document matrix using TF-iDF method")
        self.matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        self.vocab = self.tfidf_vectorizer.vocabulary_

    def get_query_vectors(self, queries: List[str]):
        """
        :param queries:
        :return: TF-iDF representations of the queries
        """
        preprocessed_queries = [self.preprocessor.preprocess_text(query) for query in queries]
        print(preprocessed_queries)
        return self.tfidf_vectorizer.transform(preprocessed_queries)#.transpose()

    def get_closeness(self, query_vectors: np.ndarray):
        """
        :param query_vectors:
        :return: array of cosine similarities between query vectors and the collection docs
        """
        return cosine_similarity(self.matrix, query_vectors)
