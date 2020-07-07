import os
from bs4 import BeautifulSoup
import nltk

class Corpus(object):
    def __init__(self):
        # self.thing = something
        self.corpus_dir = 'blogs/'
        self.filenames = self.all_files()
        self.texts = self.create_texts()
        
    def all_files(self):
        """given the corpus_dir, return the filenames in it"""
        texts = []
        for (root, _, files) in os.walk(self.corpus_dir):
            for fn in files:
                if fn[0] == '.': # a new addition!
                    pass
                else:
                    path = os.path.join(root, fn)
                    texts.append(path)
        return texts
    
    def create_texts(self):
        return [Text(filename) for filename in self.filenames]

class Text(object):
    def __init__(self, fn):
        # self.thing = something that gets you that thing
        self.filename = fn
        self.raw_html = self.get_the_text()
        self.soup = BeautifulSoup(self.raw_html, 'lxml')
        self.timestamp = self.soup.time.text
        self.p_tags = self.soup.find_all('p')
        self.text = ' '.join([tag.text for tag in self.p_tags])
        self.tokens = nltk.word_tokenize(self.soup.text)
        # TODO: include next steps in the pipeline
        # TODO: add more things based on what you want to look at
        
        
    def get_the_text(self):
        with open(self.filename, 'r') as file_in:
            return file_in.read()
         
        
    # with open(fn, 'r') as file_in:
    #     #going to open each file to READ, but idk what fin is
    #     raw_html = fin.read()
        
def main():
    # if i run the file from the terminal a la $ python3 analysis.py
    # this is what will run.
    print("hey what's up")


if __name__ == "__main__":
    main()
    
    soup = BeautifulSoup(html)
    
    
# >>> corpus.posts # give all your posts (will be a list)
# >>> first_post = corpus.posts[0]

# $ python3
# >>> import analysis
# >>> corpus = analysis.Corpus()

# if something messes up or you change something
# >>> import importlib # only has to be done once
# >>> importlib.reload(analysis)
# >>> corpus = analysis.Corpus() # re-instantiate the class




