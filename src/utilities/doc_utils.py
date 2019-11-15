
import math
import glob
import pickle

from src import configs
from src.utilities import file_utils as glfile

DOC_PREFIX = 'doc'

def get_docs_from_file(pathname):
    with open(pathname, 'r') as readfile:
        lines = readfile.readlines()
    
    return lines

def get_docs(glob_paths, clean_config, cache=True):
    results = []

    clean_config_postfix = clean_config.get_param_postfix()
    intermediary = ''.join(glob_paths)

    path_exists, path_to_file, filename, pickled_file = glfile.cache_file(DOC_PREFIX, intermediary, clean_config_postfix)

    if path_exists:
        print('\ngetting cached docs')
        return pickled_file

    for glob_path in glob_paths:
        lines = get_docs_from_file(glob_path)
        partitioned = partition_doc(lines, clean_config)
        doc = clean_doc(partitioned, clean_config)
        results.append(doc)
    
    with open(path_to_file, 'wb') as writefile:
        pickle.dump(results, writefile)

    return results

def partition_doc(text_array, clean_config):
    # skip first 1000 and last 1000 lines, publisher info and appendix.
    middle = text_array

    if len(text_array) > clean_config.start + clean_config.end:
        middle = text_array[clean_config.start:-clean_config.end]

    # take a certain percentage (slices) of the pages
    total_lines = len(text_array)
    results = []
    if clean_config.percentages and total_lines > clean_config.minimum_lines:
        for start_percentage, end_percentage in clean_config.percentages:
            start_index = math.floor(total_lines * start_percentage)
            end_index = math.floor(total_lines * end_percentage)

            results = [*results, *middle[start_index:end_index]]
    else:
        results = middle
    
    return results

def clean_doc(text_array, clean_config):
    doc = ' '.join(text_array)

    return doc

def get_author_titles(path_globs):
    results = []
    for path_glob in path_globs:
        author, title = get_author_title(path_glob)
        results.append((author, title))
    return results

def get_author_title(path):
    """
    return author, title from pathname
    input
        pathname: '/path/to/douglas, adam - hitchhiker - galaxy'

    return (
        'adam douglas',  'hitchiker galaxy'
    )
    """
    filename = path.split('/')[-1]
    filename_split = [item.strip() for item in filename.split('-')]

    author = ' '.join(filename_split[0].strip().split(',')[::-1]).strip()
    title_with_ext = ' '.join(filename_split[1:]).strip()
    title_without_ext = '.'.join(title_with_ext.split('.')[0:-1])

    return author, title_without_ext

def get_author_title_spec():
    tests = [
        ('/douglas, adam - galaxy.txt', ('adam douglas', 'galaxy')),
        ('/path/douglas, adam - galaxy.txt', ('adam douglas', 'galaxy')),
        ('/path/douglas, adam - one - galaxy.txt', ('adam douglas', 'one galaxy'))
    ]

    # get_author_title
    for index, test in enumerate(tests):
        pathname = test[0]
        expected_author = test[1][0]
        expected_title = test[1][1]
        author, title = get_author_title(pathname)
        try:
            assert author == expected_author
            assert title == expected_title
        
        except:
            print(f'\ndocs, test ERR - {index} {pathname}\n  ', author, title)

def main_test():
    get_author_title_spec()
        

if __name__ == '__main__':
    main_test()
