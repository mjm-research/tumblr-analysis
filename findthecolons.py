#find the colons!!

import nltk
file = open('FakeVoicing.txt')
#print(file.read())
#okay that worked
raw = file.read()
from nltk.tokenize import wordpunct_tokenize
tokens = wordpunct_tokenize(raw)
text = nltk.Text(tokens)
text.concordance(":")
#!!!
#but I need MORE context. like the whole post.
#or at least the location of the post
