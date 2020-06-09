import os
from bs4 import BeautifulSoup
# working with multiple files

def all_files(folder_name):
    """given a directory, return the filenames in it"""
    texts = []
    for (root, _, files) in os.walk(folder_name):
        for fn in files:
            path = os.path.join(root, fn)
            texts.append(path)
    return texts

#idk

folder = 'blogs/ghost/posts'
filenames = all_files(folder)
print(len(filenames))

#this is printing the number of total posts (aka files). 5952

all_posts = [] #[] = everything in it?
for fn in filenames: #do this thing to each file
    with open(fn, 'r') as fin:
    #i don't get this
        raw_html = fin.read()
        #does fin = each opened file?
    soup = BeautifulSoup(raw_html)
    #the soup will be the raw_html turned into less raw?
    timestamp = soup.time.text
    #my time isn't read
    p_tags = soup.find_all('p')
    post_text = ' '.join([tag.text for tag in p_tags])
    all_posts.append((timestamp,post_text))

for post in all_posts:
    print('=====')
    print(post)
    print('=====')

#what's with all the weird
