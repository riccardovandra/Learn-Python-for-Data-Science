import os #import os module

import requests #import requests module


def get_data_dir(): #Get default directory to download files

    return os.path.join(os.path.expanduser("~"), "T-Shaped Player/Development/Projects//Learn Python for Data Science/Recommendation System/") 


def create_data_dir(path): # create directory in that path if it doesn't exists

    if not os.path.isdir(path):
        os.makedirs(path)


def download(url, dest_path): #download a fle in raw content using the request module

    req = requests.get(url, stream=True)
    req.raise_for_status() # Raises stored HTTPError, if one occurred.

    #with closes the file automatically 

    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)


def get_data(data_home, url, dest_subdir, dest_filename, download_if_missing): #get the downloaded data

# if there is no parameter for data_home then create a data_dir variable from the defoult data dir and the dest_subr

    if data_home is None: #if data_home is not setted 
        data_dir = os.path.join(get_data_dir(), dest_subdir) #the data directory is the default
    else:
        data_dir = os.path.join(os.path.abspath(data_home), dest_subdir)

# create a new directory in data_dir

    create_data_dir(data_dir)

# the destination path is equal to the data directory joined by the destination filename

    dest_path = os.path.join(data_dir, dest_filename)

    if not os.path.isfile(dest_path): # if there is no file in the destination path
        if download_if_missing: #if download if missing is set to True
            download(url, dest_path) #download the file in the destination path
        else:
            raise IOError("Dataset missing.")

    return dest_path

