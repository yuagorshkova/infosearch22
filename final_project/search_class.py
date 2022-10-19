import json
import time
import numpy as np
from typing import List
from setup_logger import logger
from preprocessing import preprocessor


class Search:

    def __init__(self, data_path: str, n_docs: int):

        self.preprocessor = preprocessor

        self.corpus = None
        self.filename_index = None
        self.matrix = None
        self.vocab = None

        self.create_corpus(data_path, n_docs)

    def create_corpus(self, data_path: str, n_docs: int):
        """
        :param data_path: path to .jsnonl file with data
        :param n_docs: number of docs to process
        :return: Create a list of documents containing preprocessed texts, where each document
        corresponds to a file in the collection.
        """
        corpus = []
        filename_index = []
        logger.info(f"Preprocessing texts from {data_path} directory")

        with open(data_path, "r", encoding="utf-8") as f:
            json_corpus = list(f)[:n_docs]
        for i in range(n_docs):
            question = json.loads(json_corpus[i])["question"]
            answers = json.loads(json_corpus[i])["answers"]
            if answers:
                values = [a["author_rating"]["value"] for a in answers]
                best_answer_index = np.argmax([int(v) if v else 0 for v in values])
                text = answers[best_answer_index]["text"]
                corpus.append(text)
                filename_index.append(f"doc_{i}: {question}")
        self.filename_index = np.array(filename_index)
        self.corpus = corpus

    def create_inverted_index(self):
        pass

    def get_query_vectors(self, queries: List[str]):
        pass

    def get_closeness(self, query_vectors: np.ndarray):
        pass

    def find_closest_docs(self, query: str, n: int = 5):
        """
        :param query:
        :param n: number of result docs to list
        :return: n most similar docs
        """
        query_vector = self.get_query_vectors([query])
        distances = self.get_closeness(query_vector).ravel()
        sorted_indices = np.argsort(distances)[:-n-1:-1]

        return self.filename_index[sorted_indices]

    # можно было бы декоратор, но мне нужно возвращать значение, а не просто печатать
    def time_find_closest_docs(self, query: str, n: int = 5):
        """Same as find_closest_docs but also times it"""
        start_time = time.perf_counter()
        search_result = self.find_closest_docs(query, n)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        return search_result, total_time
