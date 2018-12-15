# -*- coding: utf-8 -*-

import os
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

if __name__ == '__main__':
    from distutils.core import setup
    extra_files = package_files('pyFuzzyME')
    setup(name='pyFuzzyME',
          packages=['pyFuzzyME'],
          zip_safe=False)
