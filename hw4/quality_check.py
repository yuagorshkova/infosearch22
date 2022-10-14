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
import random
import numpy as np


def check_quality(search, test_queries, queries_idx, top_n):


    qv = search.get_query_vector(test_queries)
    results = search.get_closeness(qv)
    top_res_idx = results.argsort(axis=0)[:-top_n-1:-1]
    results = np.take_along_axis(results, top_res_idx, axis=0)
    results = ((results - queries_idx) == 0).sum(axis=0) #есть ли нужные индексы в каждой колокне
    results = results.astype(bool).sum() #сколько правильных
    return results / queries_idx.shape[0]

# здесь много кода, который надо было раньше вынести в функции, но я не успеваю :(
#код почти полностью повторяет мэйны из 3 и 4 дз


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for the docs closest to the given query")
    parser.add_argument('-use_precalculated', action='store_true')
    parser.add_argument('-dont_use_precalculated', dest='use_precalculated', action='store_false')
    parser.set_defaults(use_precalculated=True)
    parser.add_argument("-n_queries", type=int, default=5000, help="number of queries to calculate quality on")
    parser.add_argument("-top_n", type=int, default=5, help="top n result docs to compare to query")

    config = configparser.ConfigParser()
    config.read("config.ini")

    inverted_index_path = config["paths_to_files"]["inverted_index_file_path"]

    args = parser.parse_args()

    #здесь для берта
    if not os.path.exists(inverted_index_path):
        logger.info("Existing BERT inverted index not found, creating a new one")
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
        logger.info("Finished creating BERT inverted index")
    else:
        logger.info("Using existing BERT inverted index")
        inverted_index = load_pickle(inverted_index_path)

    #здесь для BM25
    BM25_inverted_index_path = config["paths_to_files"]["BM25_inverted_index_file_path"]
    if not os.path.exists(BM25_inverted_index_path):
        logger.info("Existing BM25 inverted index not found, creating a new one")
        data_path = config["paths_to_files"]["data_path"]
        ndocs = int(config["int_constants"]["ndocs"])

        BM25_inverted_index = InvertedIndex25()
        BM25_inverted_index.create_corpus(data_path, ndocs)
        BM25_inverted_index.create_inverted_index()
        save_pickle(BM25_inverted_index, BM25_inverted_index_path)
        logger.info("Finished creating BM25 inverted index")
    else:
        logger.info("Using existing BM25 inverted index")
        BM25_inverted_index = load_pickle(BM25_inverted_index_path)

    search25 = Search25(BM25_inverted_index)
    searchBERT = Search(inverted_index)

    n_queries = args.n_queries
    top_n = args.top_n
    queries_idx = np.array(random.sample(range(len(inverted_index.filename_index)), n_queries))
    queries = [''.join(text.split(':')[1:])for text in inverted_index.filename_index[queries_idx]]

    quality_bert = check_quality(searchBERT, queries, queries_idx, top_n)
    print(quality_bert)
    quality_bm25 = check_quality(search25, queries, queries_idx, top_n)
    print(quality_bm25)


