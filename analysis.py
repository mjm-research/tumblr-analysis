import os
from bs4 import BeautifulSoup
import nltk
import re
from nltk.util import ngrams
import pandas as pd
from datetime import datetime

class Corpus(object):
    def __init__(self):
        # self.thing = something
        self.corpus_dir = 'blogs/'
        self.filenames = self.all_files()
        self.metadata = pd.read_csv('metadata.csv')
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.texts = self.sort_by_date(self.create_texts())
        self.tilde_posts = [text for text in self.texts if text.contains_tilde]
        # TODO: DISCUSS get a subset of the posts for a particular blog or other types of metadata
        # interested across time and across blogs
    
    def get_subset_by_metadata(self, key, value):
        """sub_corpus = this_corpus.get_subset_by_metadata('blog','ghost')"""
        return [text for text in self.texts if getattr(text, key) == value]
        
    def find_text(self, fn):
        for text in self.texts:
            if text.filename.split('/')[-1] == fn:
                return text
    
    def function_name(parameters_for_the_function):
        print('the stuff here')
        return none    
        
    def sort_by_date(self, texts):
        # Start here - convert the timestamp into datetime objects and sort off date.
        texts.sort(key=lambda x: x.timestamp)
        return texts
            
    def get_particular_blog(self, search_term):
        # TODO - make sure this works when we have metadata
        return [text for text in self.texts if text.post_metadata['blog'] == search_term]
    
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
        return [Text(filename, self.stopwords, self.metadata) for filename in self.filenames]

class Text(object):
    def __init__(self, fn, stopwords, metadata):
        # self.thing = something that gets you that thing
        self.filename = fn
        self.blog = self.filename.split('/')[1]
        self.post_metadata = metadata.loc[metadata['blog'] == self.blog]
        # TODO: fill in the metadata based on the stuff in the spreadsheet
        # Note: this will dynamically assign attributes based on your metadata.
        for item in self.post_metadata:
            setattr(self, item, self.post_metadata[item].iloc[0])
        # self.composite = self.post_metadata['composite']
        self.raw_html = self.get_the_text()
        self.soup = BeautifulSoup(self.raw_html, 'lxml')
        self.time_as_string =self.soup.time.text
        self.timestamp = datetime.strptime(self.soup.time.text,'%m/%d/%Y %H:%M:%S %p')
        self.p_tags = self.soup.find_all('p')
        self.text = ' '.join([tag.text for tag in self.p_tags])
        self.post_tags = [tag.text for tag in self.soup.find_all('a')][:-1]
        self.note_count = self.find_note_count()
        self.tokens = nltk.word_tokenize(self.text)
        self.stopwords = stopwords
        self.freq_dist = nltk.FreqDist(self.tokens)
        self.contains_tilde = self.tilde_true()
        self.quoted_material = self.find_quoted_material()
        if self.contains_tilde:
            # TODO: see if this causes problems downstream. if it does, you can just take the if statement out.
            self.tilde_paragraphs = [p.text for p in self.p_tags if '~' in p.text]
            self.tilde_counts = self.count_tildes()
        # TODO: can lexical diversity be more than 1? 
        self.lexical_diversity = len(self.tokens)/len(list(set(self.tokens)))
        # all words divided by unique words
        self.irreg_cap = self.find_irreg_cap()
    def find_quoted_material(self):
        # TODO: START HERE NEXT TIME
        pass
    
    def count_tildes(self):
        beginning_tilde_counts = len([token for token in self.tokens if token.startswith('~')])
        end_tilde_counts = len([token for token in self.tokens if token.endswith('~')])
        return (beginning_tilde_counts, end_tilde_counts)
    
    def tilde_true(self):
        # TODO: maybe refactor
        result = False
        for token in self.tokens:
            if '~' in token:
                result = True
                break
        return result
    
    def find_irreg_cap(self):
	    for token in self.tokens:
		    if token[1:].isupper()
		        result = True
	    return result
    
    def collocations(self, n):
        """take a text and get the most common phrases of length n. for example, text.collocations(3) gives you most common phrases 3 words long"""
        return nltk.FreqDist(list(ngrams(self.tokens, n)))

    def find_note_count(self):
        target = self.soup.footer.text.split('—')[1]
        return int(re.search("[0-9]+", target).group())

        # TODO: NEXT TIME START WITH - Michelle's work from Thanksgiving and then maybe work with the metadata.
        # TODO: make sure we get rid of punctuation when we want to and keep it when we do.
        # get rid of these characters in the note text - '¶', '●', '⬀', '⬈'
        # TODO: track lexical variance
        # TODO: how do we throw away video? look for a pre tag if it exists throw it away?
        # TODO: include next steps in the pipeline
        # TODO: make sure the spaces 
        # TODO: quotations - quotation marks (just in text body though). what's inside of the quotation marks
        # TODO: Why is " turning into `` by the tokenizer?
        # TODO: novel proper nouns - writing that looks like: he is a Good Dog. Named Entity Recognition -> frequency counts?
        # TODO: dialogues - letter, colon, space
        # ex - person a: blah blah blah
        # ex - person b: blah blah blah
        # TODO some way of getting word counts dynamically from the freq_dist
        # key words: pure, wholesome, “social justice”, good, moral, ethic, time, timeline, multiverse, change, progress, nasty, gross, normal, freaks, weirdos, uncomfy, y’all, tumblr, valid, invalid, rights, “no rights”
        # text.fq - our frequency distribution
        # text.fq['multiverse']
        # corpus.fq_counts = [text.fq[word] for text in corpus.texts]
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




