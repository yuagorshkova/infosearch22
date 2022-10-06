import pickle
import os
import shutil
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


# https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    final_dir_name = zipfile.namelist()[0]

    if os.path.exists("__MACOSX") and os.path.isdir("__MACOSX"):
        shutil.rmtree("__MACOSX")

    return final_dir_name


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
