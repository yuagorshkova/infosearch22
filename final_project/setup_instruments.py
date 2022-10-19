import configparser
from bm25_search_engine import load_bm25
from tfidf_search_engine import load_tfidf
from bert_search_engine import load_bert_search

config = configparser.ConfigParser()
config.read("config.ini")

data_path = config["paths_to_files"]["data_path"]
n_docs = int(config["int_constants"]["n_docs"])

BM25_path = config["paths_to_files"]["BM25_path"]
bm25 = load_bm25(BM25_path, data_path, n_docs)

TFiDF_path = config["paths_to_files"]["TFiDF_path"]
tfidf = load_tfidf(TFiDF_path, data_path, n_docs)

bert_search_path = config["paths_to_files"]["bert_search_path"]
precalculated_embeddings_url = config["paths_to_files"]["precalculated_embeddings_url"]
filename_starts_with = config["filename_format"]["filename_starts_with"]
n_files = int(config["int_constants"]["n_npy_files"])
bert_search = load_bert_search(bert_search_path, data_path, n_docs,
                               precalculated_embeddings_url, filename_starts_with, n_files)