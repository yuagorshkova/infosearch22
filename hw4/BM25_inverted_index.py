import json
from setup_logger import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import Preprocessor


class InvertedIndex25:

    def __init__(self):

        self.preprocessor = Preprocessor()
        self.count_vectorizer = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()

        self.corpus = None
        self.filename_index = None
        self.matrix = None
        self.vocab = None

    def create_corpus(self, data_path: str, ndocs: int):
        """

        :param data_path: path to .jsnonl file with data
        :param ndocs: number of docs to process
        :return: Create a list of documents containing preprocessed texts, where each document
        corresponds to a file in the collection.
        """
        corpus = []
        filename_index = []
        logger.info(f"Preprocessing texts from {data_path} directory")

        with open(data_path, "r", encoding="utf-8") as f:
            json_corpus = list(f)[:ndocs]
        for i in range(ndocs):
            question = json.loads(json_corpus[i])["question"]
            answers = json.loads(json_corpus[i])["answers"]
            if answers:
                values = [a["author_rating"]["value"] for a in answers]
                best_answer_index = np.argmax([int(v) if v else 0 for v in values])
                text = answers[best_answer_index]["text"]
                corpus.append(self.preprocessor.preprocess_text(text))
                filename_index.append(f"doc_{i}: {text}")
        self.filename_index = np.array(filename_index)
        self.corpus = corpus

    def create_inverted_index(self):
        """
        Create a BM-25 vectorized document-term matrix.
        """
        if not self.corpus:
            logger.error("Create corpus first!")
        logger.info("Creating the term-document matrix using BM25 method")
        #

        tf = self.count_vectorizer.fit_transform(self.corpus)
        self.tfidf_vectorizer.fit(self.corpus)
        idf = self.tfidf_vectorizer.idf_
        self.vocab = self.tfidf_vectorizer.vocabulary_

        len_d = tf.sum(axis=1)
        avdl = len_d.mean()

        k = 2
        b = 0.75

        vector_of_k = np.full(idf.shape, k + 1)
        matrix = tf.multiply(idf).multiply(vector_of_k).tocsr()

        # перевожу не из dense, а из scipy.sparse.coo_matrix
        # multiply -- поэлементное перемножение, но в разыы быстрее, чем итерировать
        # памяти  coo и csr используют одинаково, я проверила

        del_element = (k * (1 - b + b * len_d / avdl))
        for i, j in zip(*matrix.nonzero()):
            matrix[i, j] /= (tf[i, j] + del_element[i, 0])
        self.matrix = matrix
