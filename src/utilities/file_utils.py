

import pickle
import hashlib
import glob
CACHED_FILE_DIR = f'src/cached_docs/'

def hasher(prefix, string, postfix):
    whole_string = prefix + string + postfix

    hash_object = hashlib.sha1(whole_string.encode())
    hex_dig = hash_object.hexdigest()

    return hex_dig


def cache_file(prefix, string, postfix, ext='.pkl'):
    hex_dig = hasher(prefix, string, postfix)

    filename = f'{prefix}_{postfix}_{hex_dig}.pkl'
    path_to_file = f'{CACHED_FILE_DIR}{filename}'

    path_exists = glob.glob(path_to_file)

    pickled_file = None

    if path_exists:
        with open(path_to_file, 'rb') as readfile:
             pickled_file = pickle.load(readfile)

    return path_exists, path_to_file, filename, pickled_file

