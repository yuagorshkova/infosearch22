import os
from setup_logger import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import Preprocessor


class InvertedIndex:

    def __init__(self):

        self.preprocessor = Preprocessor()
        self.vectorizer = TfidfVectorizer()

        self.corpus = None
        self.filename_index = None
        self.matrix = None
        self.vocab = None

    def create_corpus(self, data_directory: str):
        """
        :param data_directory: :return: Create a list of documents containing preprocessed texts, where each document
        corresponds to a file in the collection.
        """
        corpus = []
        filename_index = []
        logger.info(f"Preprocessing texts from {data_directory} directory")
        for root, _, files in os.walk(data_directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as infile:
                    if not filename.startswith("."):
                        episode_text = infile.read()
                        filename_index.append(filename)
                        corpus.append(self.preprocessor.preprocess_text(episode_text))
        self.filename_index = np.array(filename_index)
        self.corpus = corpus

    def create_inverted_index(self):
        """
        Create a tf-idf vectorized document-term matrix.
        """
        if not self.corpus:
            logger.error("Create corpus first!")
        logger.info("Creating the term-document matrix using TF-iDF vectorizer")
        self.matrix = self.vectorizer.fit_transform(self.corpus)
        self.vocab = self.vectorizer.vocabulary_
