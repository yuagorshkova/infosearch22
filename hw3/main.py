import os
import configparser
import argparse
from setup_logger import logger
from inverted_index import InvertedIndex
from search import Search
from utils import save_pickle
from utils import load_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for the docs closest to the given query")
    parser.add_argument("query", type=str, help="query to search the docs for")
    parser.add_argument("-n", type=int, default=3, help="number of docs to print")

    config = configparser.ConfigParser()
    config.read("config.ini")

    inverted_index_path = config["Paths to files"]["inverted_index_file_path"]

    if not os.path.exists(inverted_index_path):
        logger.info("Existing inverted index not found, creating a new one")
        data_path = config["Paths to files"]["data_path"]
        ndocs = int(config["Constants"]["ndocs"])

        inverted_index = InvertedIndex()
        inverted_index.create_corpus(data_path, ndocs)
        inverted_index.create_inverted_index()
        save_pickle(inverted_index, inverted_index_path)
        logger.info("Finished creating inverted index")
    else:
        logger.info("Using existing inverted index")
        inverted_index = load_pickle(inverted_index_path)

    args = parser.parse_args()
    query = args.query
    n = args.n
    search = Search(inverted_index)
    candidates = search.find_closest_docs(query, n=n)
    print(*candidates, sep="\n")
