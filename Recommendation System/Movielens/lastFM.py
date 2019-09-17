import itertools
import os
import zipfile

import numpy as np

import scipy.sparse as sp
import _common


def _read_raw_data(path):
    """
    Return the raw lines of the train and test files.
    """

    #use the zipfile module to open the 
    with zipfile.ZipFile(path) as datafile:
        # return these files, decode them and split every new line
        return (
            datafile.read("artists.dat").decode(errors="ignore").split("\n"),  #artists
            datafile.read("tags.dat").decode(errors="ignore").split("\n"),  #tags
            datafile.read("user_artists.dat").decode(errors="ignore").split("\n"), #user artists
        )


# Parse Data

def _parse(data):

    for line in data:

        if not line:
            continue

        #user_Id and item_id

        uid, iid, rating,  = [int(x) for x in line.split("\t")]

        # Subtract one from ids to shift
        # to zero-based indexing
        yield uid - 1, iid - 1, rating, timestamp

def _get_dimensions(train_data): 
    # there are 4 different values in train_data and test_data, using _ you don't need to add those 

    #why using sets ? becuase it's an unordered list of unique elements
    uids = set() #The set of unique ids
    iids = set() #The set of unique items

    #user id and item id
    for uid, iid, _, _ in itertools.chain(train_data, test_data):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return rows, cols

#this interaction matrix is created only for the rating
def _build_interaction_matrix(rows, cols, data, min_rating):

    #creates an empty LIL matrix with rows and cols
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, _ in data:
        if rating >= min_rating:
            mat[uid, iid] = rating

    return mat.tocoo() #convert the matrix to coordinate format


def _parse_item_metadata(num_items, item_metadata_raw, genres_raw):

    genres = []

    for line in genres_raw:
        if line:
            genre, gid = line.split("|")
            genres.append("genre:{}".format(genre))

    id_feature_labels = np.empty(num_items, dtype=np.object)
    genre_feature_labels = np.array(genres)

    id_features = sp.identity(num_items, format="csr", dtype=np.float32)
    genre_features = sp.lil_matrix((num_items, len(genres)), dtype=np.float32)

    for line in item_metadata_raw:

        if not line:
            continue

        splt = line.split("|")

        # Zero-based indexing
        iid = int(splt[0]) - 1
        title = splt[1]

        id_feature_labels[iid] = title

        item_genres = [idx for idx, val in enumerate(splt[5:]) if int(val) > 0]

        for gid in item_genres:
            genre_features[iid, gid] = 1.0

    return (
        id_features,
        id_feature_labels,
        genre_features.tocsr(),
        genre_feature_labels,
    )


def fetch_lastFM(
    data_home=None,
    indicator_features=True,
    genre_features=False,
    min_rating=0.0,
    download_if_missing=True,
):
    """
    Fetch the `Movielens 100k dataset <http://grouplens.org/datasets/movielens/100k/>`_.
    The dataset contains 100,000 interactions from 1000 users on 1700 movies,
    and is exhaustively described in its
    `README <http://files.grouplens.org/datasets/movielens/ml-100k-README.txt>`_.
    Parameters
    ----------
    data_home: path, optional
        Path to the directory in which the downloaded data should be placed.
        Defaults to ``~/lightfm_data/``.
    indicator_features: bool, optional
        Use an [n_items, n_items] identity matrix for item features. When True with genre_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_items, n_items + n_genres].
    genre_features: bool, optional
        Use a [n_items, n_genres] matrix for item features. When True with item_indicator_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_items, n_items + n_genres].
    min_rating: float, optional
        Minimum rating to include in the interaction matrix.
    download_if_missing: bool, optional
        Download the data if not present. Raises an IOError if False and data is missing.
    Notes
    -----
    The return value is a dictionary containing the following keys:
    Returns
    -------
    train: sp.coo_matrix of shape [n_users, n_items]
         Contains training set interactions.
    test: sp.coo_matrix of shape [n_users, n_items]
         Contains testing set interactions.
    item_features: sp.csr_matrix of shape [n_items, n_item_features]
         Contains item features.
    item_feature_labels: np.array of strings of shape [n_item_features,]
         Labels of item features.
    item_labels: np.array of strings of shape [n_items,]
         Items' titles.
    """



    #print('checking for errors....')
    if not (indicator_features or genre_features):
        raise ValueError(
            "At least one of item_indicator_features " "or genre_features must be True"
        )
    

    #print('No Errors found....proceding to download dataset')
    zip_path = _common.get_data(
        data_home,
        (
            "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
        ),
        "LastFM",
        "lastfm-2k.zip",
        download_if_missing,
    )
    #print('Dataset downloaded... proceding to reading the dataset')
    # Load raw data
    try:
        #put in this variables the data gained from the zip
        (artists_raw, tags_raw, user_artists_raw) = _read_raw_data(zip_path) 
    except zipfile.BadZipFile:
        # Download was corrupted, get rid of the partially
        # downloaded file so that we re-download on the
        # next try.
        os.unlink(zip_path)
        raise ValueError(
            "Corrupted Movielens download. Check your "
            "internet connection and try again."
        )

    # Figure out the dimensions
    num_users, num_items = _get_dimensions(_parse(train_raw), _parse(test_raw))

    # Load train interactions
    train = _build_interaction_matrix(
        num_users, num_items, _parse(train_raw), min_rating
    )
    # Load test interactions
    test = _build_interaction_matrix(num_users, num_items, _parse(test_raw), min_rating)

    assert train.shape == test.shape

    # Load metadata features
    (
        id_features,
        id_feature_labels,
        genre_features_matrix,
        genre_feature_labels,
    ) = _parse_item_metadata(num_items, item_metadata_raw, genres_raw)

    assert id_features.shape == (num_items, len(id_feature_labels))
    assert genre_features_matrix.shape == (num_items, len(genre_feature_labels))

    if indicator_features and not genre_features:
        features = id_features
        feature_labels = id_feature_labels
    elif genre_features and not indicator_features:
        features = genre_features_matrix
        feature_labels = genre_feature_labels
    else:
        features = sp.hstack([id_features, genre_features_matrix]).tocsr()
        feature_labels = np.concatenate((id_feature_labels, genre_feature_labels))

    data = {
        "train": train,
        "test": test,
        "item_features": features,
        "item_feature_labels": feature_labels,
        "item_labels": id_feature_labels,
    }

    return data