import pickle
import requests
import io
import torch
import numpy as np
from os.path import join


def download_npy(url: str):
    """
    :param url:
    :return: np.ndarray stored in the file from url
    """
    download = requests.get(url).content
    return np.load(io.BytesIO(download))


def download_npy_embeddings(directory_url: str, filename_starts_with: str, n_files: int):
    """
    :param directory_url: directory with files with embeddings
    :param filename_starts_with: filename up to counter ("embedding_" for filename "embedding_0.npy")
    :param n_files: number of npy files in the directory, all named the same with a counter at the end
    :return: 2D np.ndarray of all the embeddings
    """
    url = join(directory_url, filename_starts_with + str(0) + ".npy?raw=true")
    matrix = download_npy(url)
    for i in range(1, n_files):
        url = join(directory_url, filename_starts_with + str(i) + ".npy?raw=true")
        array_to_append = download_npy(url)
        matrix = np.vstack((matrix, array_to_append))
    return matrix


def load_pickle(filepath: str):
    """
    :param filepath:
    :return: object from .pickle file
    """
    with open(filepath, 'rb') as handle:
        new_object = pickle.load(handle)
    return new_object


def save_pickle(object_to_save, filepath: str):
    """
    :param object_to_save:
    :param filepath:
    Saves object as a .pickle file
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

