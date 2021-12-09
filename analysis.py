import os
from bs4 import BeautifulSoup
import nltk
import re
from nltk.util import ngrams
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
import cProfile
import pstats
import random
from progress.bar import Bar
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import string

class Corpus(object):
    def __init__(self):
        # self.thing = something
        self.clean_files()
        # Query options are "SB" (spongebob), "LD" (lexical diversity), "any given word" (general search queries)
        # Can also search collocations two ways - the first is to just pass "TC" to get a list of the top collocation for each file
        # Second is "CS: A Collocation" to search for a particular collocation
        # IMPORTANT FLAGS HERE
        self.query = 'SB'
        self.category = 'absolutism'
        # TODO: self.metadata_file = either metadata.csv or test-metadata.csv
        sample = False
        # self.combine_category = (False, category_name, how to merge it)
        # set sample = False if you want to run on the smaller test corpus
        if sample:
            # uncomment next two lines for very specific testing on a controlled subfolder
            self.corpus_dir = 'test-blogs/'
            self.filenames = self.all_files()
            # comment next three lines if testing on test-blogs
            # self.corpus_dir = 'test-blogs'
            # self.filenames = random.sample(self.all_files(),1000)
            self.metadata = pd.read_csv('test-metadata.csv')
        else:
            self.corpus_dir = 'real-blogs'
            # self.filenames = self.all_files() 
            # uncomment above and comment below to run on whole corpus
            self.filenames = random.sample(self.all_files(),2500)
            self.metadata = pd.read_csv('metadata.csv')
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.texts = self.sort_by_date(self.create_texts())
        # data_to_graph can take a general search term (in which case it adds the raw word counts, though we could make it average) or one of these specialized requests: ['LD', 'SB']
        # gets the raw data to graph
        self.filenames_by_query = self.get_filenames_by_query(self.query)
        self.export_filenames_by_query()
        if self.query == 'TC':
            pass
        else:
            self.data_to_graph = self.get_data_over_time(self.texts, self.query)
            # organize it into a set of dataframes, one for each category in the metadata sheet
            self.category_frames = self.divide_into_categories(self.data_to_graph)
            # regularize each by month
            self.regularized_category_frames = [self.regularize_data_frame(df, self.query) for df in self.category_frames]
            if self.combine and self.combine[0] == True:
                # call to 
                pass
            self.graph(self.regularized_category_frames)
            print('Complete! Be sure to check errors.txt for anything that might have quietly failed.')

    def combine_category(self):
        pass

    def spongebob_tester(self):
        with open('filenames_by_query.txt','r') as filein:
            return [Text(fn.strip('\n'), self.stopwords, self.metadata,'SB', self.category).get_irreg_cap_tokens() for fn in filein.readlines()]

    def clean_files(self):
        """Cleans up past files by emptying them so they can get new content from this run."""
        #TODO: check to see if these files exist
        with open('errors.txt', 'r+') as fin:
            fin.truncate(0)
        with open('filenames_by_query.txt', 'r+') as fin:
            fin.truncate(0)
        with open('empty_files.txt', 'r+') as fin:
            fin.truncate(0)


    def divide_into_categories(self, df):
        """takes a single dataframe, tagged for categories, and divides into separate dataframes based on each type"""

        return [df[df.CATEGORY == this_category] for this_category in set(df.CATEGORY.values)]

    def get_filenames_by_query(self, query):
        if query == 'SB':
            return [text.filename for text in self.texts if text.irreg_cap]
        elif query == '~':
            return [text.filename for text in self.texts if text.contains_tilde]
        elif query == 'DIALOGUE':
            return [text.filename for text in self.texts if text.contains_dialogue]
        elif query == "TC":
            return [(text.filename, text.most_common_collocation) for text in self.texts]
        elif query.startswith("CS:"):
            # break the collocation query into something we can search for
            proc_query = (self.query.lower().split()[1], self.query.lower().split()[2])
            print(proc_query)
            return [text.filename for text in self.texts if text.collocation_freq_dist[proc_query]]
        else:
            return [text.filename for text in self.texts if text.freq_dist[query]]

    def export_filenames_by_query(self):
        with open('filenames_by_query.txt', 'w') as fout:
            for line in self.filenames_by_query:
                fout.write(str(line) + '\n')

    def graph(self, dataframes):
        """given a set of dataframes to graph, graph them"""
        plt.style.use('seaborn-whitegrid')
        # should be five colors bc should be five categories
        colors = {0:'b', 1: 'g', 2: 'r', 3: 'm', 4: 'y', 5: 'c', 6: 'k'}
        color_count = 0
        for dataframe in dataframes:
            plt.plot(dataframe['DATE'],dataframe['DATA'].values, color=colors[color_count],label=dataframe.CATEGORY.iloc[0])
            color_count += 1
        plt.legend()
        plt.xticks(rotation=90)
        if self.query == 'SB':
            query = 'SpongeBob Meme'
        if self.query == 'DIALOGUE':
            query = 'Dialogue'
        if self.query == '~':
            query = 'Q' 
        else:
            query = self.query
        plt.title('Searching for ' + query + ' by ' + self.category)
        plt.show()

    def get_data_over_time(self, texts, query):
        df = pd.DataFrame()
        df['CATEGORY'] = [getattr(text, text.category) for text in texts]
        df['DATE'] = [datetime.strptime(text.timestamp.strftime('%Y-%m'),'%Y-%m') for text in texts]
        if query == 'LD':
            df['DATA'] = [text.lexical_diversity for text in texts]
        elif query == '~':
            df['DATA'] = [text.contains_tilde for text in texts]
        elif query == 'SB':
            df['DATA'] = [text.irreg_cap for text in texts]
        elif query.startswith('CS:'):
            proc_query = (self.query.lower().split()[1], self.query.lower().split()[2])
            df['DATA'] = [text.collocation_freq_dist[proc_query] for text in texts]
        elif query == 'DIALOGUE':
            df['DATA'] = [text.contains_dialogue for text in texts]
        # elif query == 'COLLOCATIONS':
                # df['DATA'] = [text.collocations for text in corpus.texts]
        elif query:
            df['DATA'] = [text.freq_dist[query] for text in texts]
        else:
            raise NameError('No query given')
        return df

    def regularize_data_frame(self, df, query):
        # averages data per month - you'll want to make sure it separates out different blog types here
        # we might want to add raw counts rather than averaging them sometimes
        unique_dates = set(df.DATE.values)
        converted_df = pd.DataFrame()
        converted_df['DATE'] = [date for date in unique_dates]
        converted_df['CATEGORY'] = df.CATEGORY.iloc[0]
        if query == 'LD':
            # only average LD for now, otherwise add them
            converted_df['DATA'] = [df[df['DATE'] == date]['DATA'].mean() for date in unique_dates]
        else:
            converted_df['DATA'] = [df[df['DATE'] == date]['DATA'].sum() for date in unique_dates]
        return converted_df.sort_values(by=['DATE'])
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
        text_list = []
        bar = Bar('Processing', max=len(self.filenames))
        for filename in self.filenames:
            try:
                text_list.append(Text(filename, self.stopwords, self.metadata, self.query, self.category))
                bar.next()
            except AttributeError:
                with open('empty_files.txt', 'a') as fout:
                    fout.write(filename + '\n')
                bar.next()
            except Exception:
                with open('errors.txt', 'a') as fout:
                    fout.write('Failed on ' + filename +'\n')
                    fout.write(str(traceback.format_exc())+'\n')
                    fout.write('====='+'\n')
                bar.next()
        bar.finish()
        return text_list

class Text(object):
    def __init__(self, fn, stopwords, metadata, query, category):
        # self.thing = something that gets you that thing
        self.filename = fn
        self.category = category
        self.blog = self.filename.split('/')[1]
        # using the blog name, look at the metadata csv and grab the rest of the metadata
        self.post_metadata = metadata.loc[metadata['blog'] == self.blog]
        # this will dynamically assign attributes based on your metadata.
        for item in self.post_metadata:
            setattr(self, item, self.post_metadata[item].iloc[0])
        with open(fn, 'r') as fin:
            #going to open each file to READ, but idk what fin is
            self.soup = BeautifulSoup(fin.read(), 'lxml')
        self.time_as_string =self.soup.time.text
        if self.time_as_string.endswith('PM') or self.time_as_string.endswith('AM'):
            self.timestamp = datetime.strptime(self.soup.time.text,'%m/%d/%Y %H:%M:%S %p')
        else:
            self.timestamp = datetime.strptime(self.soup.time.text,'%m/%d/%Y %H:%M:%S')
        # try:
        #     self.timestamp = datetime.strptime(self.soup.time.text,'%m/%d/%Y %H:%M:%S %p')
        # except ValueError:
        #     # koreanqueer formats their timestamps differently for some reason
        self.p_tags = self.soup.find_all('p')
        self.text = ' '.join([tag.text for tag in self.p_tags])
        # self.tokens = nltk.tokenize.WordPunctTokenizer().tokenize(self.text) # would be faster but break apart tilde words
        self.tokens = nltk.word_tokenize(self.text)
        self.cleaned_tokens = self.clean_tokens()
        self.freq_dist = nltk.FreqDist(self.tokens)
        if query == '~':
            self.contains_tilde = self.tilde_true()
            if self.contains_tilde:
                self.tilde_paragraphs = [p.text for p in self.p_tags if '~' in p.text]
                self.tilde_counts = self.count_tildes()
        elif query == 'LD':
            self.lexical_diversity = len(set(self.tokens))/len(self.tokens)
        elif query == 'SB':
            self.irreg_cap = self.find_irreg_cap()
            if self.irreg_cap:
                self.irreg_cap_tokens = self.get_irreg_cap_tokens()
        elif query == 'DIALOGUE':
            self.contains_dialogue = self.find_dialogue()
        elif query == 'COLLOCATIONS' or query == 'TC' or query.startswith('CS:'):
            self.most_common_collocation = self.find_collocations()
            self.collocation_freq_dist = nltk.FreqDist(list(nltk.bigrams(self.cleaned_tokens)))
            if len(self.most_common_collocation) >= 1:
                self.top_collocation_count = self.collocation_freq_dist[self.most_common_collocation[0]]
            else:
                self.top_collocation_count = 0
        elif query == 'MISC':
            """gathers a bunch of miscellaneous attributes that we might want to have access to but are not actively used rn.
            will want to pull them out into a particular query to activate them"""
            self.post_tags = [tag.text for tag in self.soup.find_all('a')][:-1]
            self.note_count = self.find_note_count()


    def find_collocations(self):
        """finds collocations"""
        bcf = BigramCollocationFinder.from_words(self.cleaned_tokens)
        return bcf.nbest(BigramAssocMeasures.likelihood_ratio, 1)

    def clean_tokens(self):
        """cleans tokens according to the parameters we want. this is where you'll want to adjust it if it's too aggressive"""
        regex = re.compile(r'¶|●|[0-9]+\/[0-9]+\/[0-9]+|[0-9]+:[0-9]+:[0-9]+|⬈|⬀|AM|PM|’')
        # remove random junk from tumblr
        cleaned_tokens = [token for token in self.tokens if not regex.match(token)]
        # remove punctuation
        cleaned_tokens = [token.lower() for token in cleaned_tokens if token not in string.punctuation]
        return cleaned_tokens

    def find_dialogue(self):
        """looks to see if a post contains dialogue type characters and, if yes, return the filename"""
        result = False
        if self.freq_dist['\''] or self.freq_dist['\"'] or self.freq_dist[':'] or self.freq_dist['\'\'']:
            result = True
        return result
        # old method for searching (regex needs work)
        # search = r'\w+: \w+|\"\w+ \w+\"|\'\w+: \w+|\''
        # p_result = []
        # for p in self.p_tags:
        #     if re.findall(search, p.text):
        #         p_result.append(p.text)
        #
        #return p_result

    def count_tildes(self):
        beginning_tilde_counts = len([token for token in self.tokens if token.startswith('~')])
        end_tilde_counts = len([token for token in self.tokens if token.endswith('~')])
        return (beginning_tilde_counts, end_tilde_counts)

    def tilde_true(self):
        # TODO: maybe refactor by looking at the FreqDist instead of token by token. problem right now is
        # that the tokenizer won't split on ~
        # result = False
        #     if self.freq_dist['~']:
        #         result = True
        # return result
        result = False
        for token in self.tokens:
            if '~' in token:
                result = True
                break
        return result

    def find_irreg_cap(self):
        result = False
        for token in self.tokens:
            if self.validate_irreg_caps(token):
                result = True
                break
        return result

    def get_irreg_cap_tokens(self):
        p_results = []
        for token in self.tokens:
            if self.validate_irreg_caps(token):
                p_results.append(token)
            #  
        return p_results

    def validate_irreg_caps(self, token):
        return len(token) > 1 and not token.isupper() and not token[1:].islower() and not re.match('[0-9]{1,4}\/[0-9]{1,2}\/[0-9]{1,2}|[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}|[0-9]+', token) and not any(not c.isalnum() for c in token) and token not in nltk.corpus.names.words() and self.count_percentage_of_caps(token) > 0.3 and token not in ['TERFs', 'IDs', 'MP3s', 'PayPal', 'PoC', 'PCs', 'NPCs']

    def common_ngrams(self, n):
        """take a text and get the most common phrases of length n. for example, text.collocations(3) gives you most common phrases 3 words long"""
        return nltk.FreqDist(list(ngrams(self.tokens, n)))

    def find_note_count(self):
        try:
            target = self.soup.footer.text
            return int(re.search("[0-9]+", target).group())
        except:
            return 0


    def get_the_text(self):
        with open(self.filename, 'r') as file_in:
            return file_in.read()
    
    def count_percentage_of_caps(self, word):
        result = sum(1 for c in word if c.isupper())/len(word)
        return result


def main():
    # if i run the file from the terminal a la $ python3 analysis.py
    # this is what will run.
    time_program()


if __name__ == "__main__":
    main()



# $ python3
# >>> import analysis
# >>> corpus = analysis.Corpus()

# if something messes up or you change something
# >>> import importlib # only has to be done once
# >>> importlib.reload(analysis)
# >>> corpus = analysis.Corpus() # re-instantiate the class
# >>> corpus.texts[0].raw_text # to get the raw text for text number 1

# Use the attributes in conjunction with each other to find things you care about and where to look for them. For example:
# >>> for text in corpus.texts:
# ...     if text.dialogue:
# ...             print(text.dialogue)
# ...             print(text.filename)
# ...             print(text.time_as_string)


#   (to pull out just a single file in python)
# import analysis
# import nltk
# import pandas as pd
# test = analysis.Text('real-blogs/bripopsicle/posts/28782550789.html', nltk.corpus.stopwords.words('english'),pd.read_csv('test-metadata.csv'),'SB', 'absolutism')
#    (where the only thing you should need to modify above is the filename)

# Essential
# TODO: Document the places you might especially want to make changes (brandon will do)
# TODO: Michelle is going to working on metadata more and make sure we don't need anything else. 
# TODO: when metadata is done, move everything to Jackie's cloud (last)



# Stretch 
# HOLD: novel proper nouns - writing that looks like: he is a Good Dog. Named Entity Recognition -> frequency counts?
# HOLD: reading for particular styles - reading for youth voices, innocent youth, old person standard English, academic. intermingling the different registers within a single post
# HOLD: Create better collocation that searches without looking at categories collapses all authors for a week or month together
# Hold: Troubleshoot "math domain error" in real-blogs/achtervulgan315/posts/178376711031.html and similar files - https://github.com/nltk/nltk/issues/2200 

