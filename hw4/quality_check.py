import os
import configparser
import argparse
from setup_logger import logger
from inverted_index import InvertedIndex
from BM25_inverted_index import InvertedIndex25
from search import Search
from BM25_search import Search25
from utils import save_pickle
from utils import load_pickle

def check_quality(inverted_index, search):


# здесь много кода, который надо было раньше вынести в функции, но я не успеваю :(
#код почти полностью повторяет мэйны из 3 и 4 дз
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for the docs closest to the given query")
    parser.add_argument("query", type=str, help="query to search the docs for")
    parser.add_argument("-n", type=int, default=3, help="number of docs to print")
    parser.add_argument("-use_precalculated", type=bool, default=True, help="download and use precalculated embeddings")

    config = configparser.ConfigParser()
    config.read("config.ini")

    inverted_index_path = config["paths_to_files"]["inverted_index_file_path"]

    args = parser.parse_args()

    #здесь для берта
    if not os.path.exists(inverted_index_path):
        logger.info("Existing inverted index not found, creating a new one")
        data_path = config["paths_to_files"]["data_path"]
        ndocs = int(config["int_constants"]["ndocs"])

        inverted_index = InvertedIndex()
        inverted_index.create_corpus(data_path, ndocs)

        use_precalculated = args.use_precalculated
        if use_precalculated:
            precalculated_embeddings_url = config["paths_to_files"]["precalculated_embeddings_url"]
            filename_starts_with = config["filename_format"]["filename_starts_with"]
            n_files = int(config["int_constants"]["n_npy_files"])
            inverted_index.create_inverted_index(precalculated_embeddings_url,
                                                 filename_starts_with,
                                                 n_files)
        save_pickle(inverted_index, inverted_index_path)
        logger.info("Finished creating inverted index")
    else:
        logger.info("Using existing inverted index")
        inverted_index = load_pickle(inverted_index_path)

    #здесь для BM25
    BM25_inverted_index_path = config["paths_to_files"]["BM25_inverted_index_file_path"]
    if not os.path.exists(BM25_inverted_index_path):
        logger.info("Existing inverted index not found, creating a new one")
        data_path = config["paths_to_files"]["data_path"]
        ndocs = int(config["int_constants"]["ndocs"])

        BM25_inverted_index = InvertedIndex25()
        BM25_inverted_index.create_corpus(data_path, ndocs)
        BM25_inverted_index.create_inverted_index()
        save_pickle(BM25_inverted_index, BM25_inverted_index_path)
        logger.info("Finished creating inverted index")
    else:
        logger.info("Using existing inverted index")
        BM25_inverted_index = load_pickle(BM25_inverted_index_path)

    #