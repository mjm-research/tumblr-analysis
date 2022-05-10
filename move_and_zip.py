"""given a folder name:
take all folders in it, grab only the posts folder in them
Move them to another location
Zip them up there"""

import os
import tarfile

def all_files(corpus_dir):
        """given the corpus_dir, return the filenames in it"""
        texts = []
        for (root, _, files) in os.walk(corpus_dir):
            for fn in files:
                if fn[0] == '.':
                    pass
                else:
                    path = os.path.join(root, fn)
                    print(os.path.splitext(path))
                    texts.append(path)
        return texts

def throw_away_files(files):
    """throw away the filenames we don't care about"""
    return [file for file in files if os.path.splitext(file)[0].split('/')[2] == 'posts']

def create_tar(files, destination):
    """create a tar from a list of files"""
    with tarfile.open(destination, 'w') as archive:
        for file in files:
            archive.add(file)

def main():
    corpus_dir = 'real-blogs'
    destination = 'compressed-blogs.tar.gz'
    fns = all_files(corpus_dir)
    # fns = throw_away_files(fns)
    # create_tar(fns, destination)
    # if i run the file from the terminal a la $ python3 analysis.py
    # this is what will run.


if __name__ == "__main__":
    main()
