import os
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from setup_logger import logger
from setup_bert import get_sentences_vectors
from setup_bert import TOKENIZER
from setup_bert import MODEL
from utils import save_pickle
from utils import load_pickle
from utils import download_npy_embeddings
from search_class import Search


def load_bert_search(bert_search_path, data_path, n_docs,
                     directory_url, filename_starts_with, n_files):
    if not os.path.exists(bert_search_path):
        logger.info("Existing BERT inverted index not found, creating a new one")
        bert_search = BertSearch(data_path, n_docs, directory_url, filename_starts_with, n_files)
        save_pickle(bert_search, bert_search_path)
        logger.info("Finished creating BERT search engine")
    else:
        logger.info("Using existing BERT inverted index")
        bert_search = load_pickle(bert_search_path)
    return bert_search


class BertSearch(Search):

    def __init__(self, data_path: str, n_docs: int,
                 directory_url: str, filename_starts_with: str, n_files: int):
        super().__init__(data_path, n_docs)
        self.tokenizer = TOKENIZER
        self.model = MODEL

        self.precalculated_embeddings_url = directory_url
        self.filename_starts_with = filename_starts_with
        self.n_files = n_files

        self.create_inverted_index()

    def create_inverted_index(self):
        """
        Create a bert vectorized document-term matrix.
        """
        if not self.corpus:
            logger.error("Create corpus first!")

        logger.info("Downloading precalculated embeddings")
        embeddings = download_npy_embeddings(self.precalculated_embeddings_url,
                                             self.filename_starts_with,
                                             self.n_files)
        self.matrix = embeddings

    def get_query_vectors(self, queries: List[str]):
        """
        :param queries:
        :return: BERT representations of the queries
        """
        preprocessed_queries = [self.preprocessor.preprocess_text(query) for query in queries]
        print(preprocessed_queries)
        return get_sentences_vectors(preprocessed_queries,
                                     self.tokenizer,
                                     self.model)

    def get_closeness(self, query_vectors: np.ndarray):
        """
        :param query_vectors:
        :return: array of cosine similarities between query vectors and the collection docs
        """
        return cosine_similarity(self.matrix, query_vectors)
