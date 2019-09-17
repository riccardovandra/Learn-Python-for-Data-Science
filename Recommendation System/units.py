# Module for downloading, reading and parsing a file
import os
import requests


def get_default_path():
    return os.path.join(os.path.expanduser('~'), 'T-Shaped Player/Development/Projects/Learn Python for Data Science/Recommendation System/')


def download_raw(url, dest_path):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, 'wb') as file:
        # this is used in order to avoid downloading and streaming the whole document at once
        for chunk in req.iter_content(chunk_size=2 ** 20):
            file.write(chunk)


url1 = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'


print(get_default_path())
