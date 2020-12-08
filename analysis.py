import os
from bs4 import BeautifulSoup
import nltk
import re
from nltk.util import ngrams
import pandas as pd
from datetime import datetime
import re

class Corpus(object):
    def __init__(self):
        # self.thing = something
        self.corpus_dir = 'blogs/'
        self.filenames = self.all_files()
        self.metadata = pd.read_csv('metadata.csv')
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.texts = self.sort_by_date(self.create_texts())
        self.tilde_posts = [text for text in self.texts if text.contains_tilde]
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
        # this will dynamically assign attributes based on your metadata.
        for item in self.post_metadata:
            setattr(self, item, self.post_metadata[item].iloc[0])
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
        if self.irreg_cap:
            self.irreg_cap_paragraphs = self.get_paragraphs_with_filter("token[1:].isupper() and token not in ['AM', 'PM']")
        self.dialogue = self.find_dialogue()
        # example of how to use the filter function:
        # self.paras_starting_with_cap_a = self.get_paragraphs_with_filter("token[0] == 'A'")
    def find_quoted_material(self):
        # use regex101.com to help build the regex tester
        # TODO: Michelle might be able to do this
        # TODO: START HERE NEXT TIME
        # TODO: quotations - quotation marks (just in text body though). what's inside of the quotation marks

        pass
    
    def find_dialogue(self):
        search = r'\w+: \w+'
        p_result = []
        for p in self.p_tags:
            if re.findall(search, p.text):
                p_result.append(p.text)
        
        return p_result
        
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
        result = False
        for token in self.tokens:
            if token[1:].isupper() and token not in ['AM', 'PM']:
                result = True
                break
        return result
        
    def get_paragraphs_with_filter(self, expression):
        """call this on a text to get paragraphs using a particular filter on the interior tokens. for example -
        self.get_paragraphs_with_filter("token[1:].isupper()")
        self.get_paragraphs_with_filter("token[0] == 'A'")
        returns all the spongebob paragraphs (they will contain tokens where an interior character is capitalized)
        """
        p_results = []
        for p in self.p_tags:
            for token in nltk.word_tokenize(p.text):
                if eval(expression):
                    p_results.append(p)
        return p_results

    def collocations(self, n):
        """take a text and get the most common phrases of length n. for example, text.collocations(3) gives you most common phrases 3 words long"""
        return nltk.FreqDist(list(ngrams(self.tokens, n)))

    def find_note_count(self):
        target = self.soup.footer.text.split('—')[1]
        return int(re.search("[0-9]+", target).group())


        # TODO: novel proper nouns - writing that looks like: he is a Good Dog. Named Entity Recognition -> frequency counts?
        # TODO: make sure we get rid of punctuation when we want to and keep it when we do.
        # get rid of these characters in the note text - '¶', '●', '⬀', '⬈'
        # TODO: track lexical variance
        # TODO: how do we throw away video? look for a pre tag if it exists throw it away?
        # TODO: include next steps in the pipeline
        # TODO: make sure the spaces 
        # TODO: Why is " turning into `` by the tokenizer?
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

# Use the attributes in conjunction with each other to find things you care about and where to look for them. For example:
# >>> for text in corpus.texts:
# ...     if text.dialogue:
# ...             print(text.dialogue)
# ...             print(text.filename)
# ...             print(text.time_as_string)


#   (to pull out just the test file in python)
# import analysis
# this_corpus = analysis.Corpus()
# test_corpus = this_corpus.get_subset_by_metadata('blog','test')


# TODO: in may or june try running this on the whole corpus? in addition to what that remains.