import json
from setup_logger import logger
import numpy as np
from setup_model import TOKENIZER, MODEL, get_sentences_vectors
from preprocessing import Preprocessor
from utils import download_npy_embeddings


class InvertedIndex:

    def __init__(self):

        self.preprocessor = Preprocessor()
        self.tokenizer = TOKENIZER
        self.model = MODEL

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

    def create_inverted_index(self,
                              precalculated_embeddings_url: str = None,
                              filename_starts_with: str = None,
                              n_files: int = None):
        """
        Create a BERT vectorized document matrix.
        """
        if not self.corpus:
            logger.error("Create corpus first!")
        logger.info("Getting the corpus vectorized with BERT")

        if precalculated_embeddings_url:
            logger.info("Downloading precalculated embeddings")
            embeddings = download_npy_embeddings(precalculated_embeddings_url,
                                                 filename_starts_with,
                                                 n_files)
            self.matrix = embeddings
        else:
            logger.info("Calculating embeddings from scratch")
            self.matrix = get_sentences_vectors(self.corpus, self.tokenizer, self.model)
