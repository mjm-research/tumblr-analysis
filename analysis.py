import os
from bs4 import BeautifulSoup
import nltk
import re

class Corpus(object):
    def __init__(self):
        # self.thing = something
        self.corpus_dir = 'blogs/'
        self.filenames = self.all_files()
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.texts = self.create_texts()
        # TODO: get a subset of the posts for a particular blog
        # TODO: implement a metadata csv file
        
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
        return [Text(filename, self.stopwords) for filename in self.filenames]

class Text(object):
    def __init__(self, fn, stopwords):
        # self.thing = something that gets you that thing
        self.filename = fn
        # TODO: self.blog = the actual blog the post comes from
        self.raw_html = self.get_the_text()
        self.soup = BeautifulSoup(self.raw_html, 'lxml')
        self.timestamp = self.soup.time.text
        self.p_tags = self.soup.find_all('p')
        self.text = ' '.join([tag.text for tag in self.p_tags])
        self.post_tags = [tag.text for tag in self.soup.find_all('a')][:-1]
        self.note_count = self.find_note_count()
        self.tokens = nltk.word_tokenize(self.text)
        self.stopwords = stopwords
        self.freq_dist = nltk.FreqDist(self.tokens)
        
    def find_note_count(self):
        target = self.soup.footer.text.split('—')[1]
        return int(re.search("[0-9]+", target).group())

        # get rid of these characters in the note text - '¶', '●', '⬀', '⬈'

        # TODO: how do we throw away video? look for a pre tag if it exists throw it away?
        # TODO: include next steps in the pipeline
        # TODO: make sure the spaces 
        # TODO: tildes - ~word word~ for ex. ~social justice~. or a ~ on one side, either front or back. probably interested in the whole post or the whole sentence. also interested in tilde counts
        # TODO: quotations - quotation marks (just in text body though). what's inside of the quotation marks
        # TODO: novel proper nouns - writing that looks like: he is a Good Dog. Named Entity Recognition -> frequency counts?
        # TODO: dialogues - letter, colon, space
        # ex - person a: blah blah blah
        # ex - person b: blah blah blah
        # TODO some way of getting word counts dynamically from the freq_dist
        # key words: pure, wholesome, “social justice”, good, moral, ethic, time, timeline, multiverse, change, progress, nasty, gross, normal, freaks, weirdos, uncomfy, y’all, tumblr, valid, invalid, rights, “no rights”
        # text.fq - our frequency distribution
        # text.fq['multiverse']
        # corpus.fq_counts = [text.fq[word] for text in corpus.texts]
        # TODO Repeated phrases? Possible to look for most common 3, 5, and 7 words in a row?
        # text.bigrams()
        # ngrams larger than three https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-corpus-of-txt-files-and-t
        # TODO: reading for particular styles - reading for youth voices, innocent youth, old person standard English, academic. intermingling the different registers within a single post
        # two approaches -  machine learning way. maybe at the sentence level or the paragraph level. topic modeling?
    
                
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
# >>> corpus[0].raw_text # to get the raw text for text number 1




