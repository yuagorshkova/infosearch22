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
    print(f"Data extracted to {final_dir_name}")

    if os.path.exists("__MACOSX") and os.path.isdir("__MACOSX"):
        shutil.rmtree("__MACOSX")

    return final_dir_name



