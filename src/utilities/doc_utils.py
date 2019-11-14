
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
