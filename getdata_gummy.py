#extract the relevant data like post text, date, etc
#saving text files, cleaning up output, doing things w/ nltk

import os
from bs4 import BeautifulSoup
#website says to do this and in brandon's code

def all_files(folder_name):
    texts = []
    for (root, _, files) in os.walk(folder_name):
        for fn in files:
            path = os.path.join(root, fn)
            texts.append(path)
    return texts

folder = 'blogs/ghost/posts'
#does .py file have to be in the same folder as ghost (e.g DHproject)
filenames = all_files(folder)
#named all the posts in the folder, fileneames
all_posts = []
#all_posts is gonna be all the content of filenames
for fn in filenames:
    #just means do it to each file i think
    with open(fn, 'r') as fin:
        #going to open each file to READ, but idk what fin is
        raw_html = fin.read()
        #raw_html is just all the info in the files being read
        #in theory, should print all the info. let's try!
        #test w/ print(raw_html)
        #okay! that is what it does, but only if indent
        #it just does one post if all the way left
        soup = BeautifulSoup(raw_html)
        #let's see what this does!
        #test: print(soup)
        #this just seemed to remove the paragraph breaks?
        timestamp = soup.time.text
        #this will show date?
        #test: print(timestamp)
        #Yes! but they aren't in order? weird
        #okay now i think this next bit is gonna remove paragraphs
        p_tags = soup.find_all('p')
        post_text = ' '.join([tag.text for tag in p_tags])
        #test: print(post_text)
        #but does include data even without specify timestamp
        #! i wanna try finding tags in 'a'
        tumblrtags = soup.find_all('a')
        tags_text = ' '.join([tag.text for tag in tumblrtags])
        #print(tags_text)
        #woo! this just adds usernames as well nbd
        all_posts.append((post_text,tags_text))
        #this makes it all weird and close together donut like
        #HOW DO I SEND THIS SOMEWHERE?? Like make a file of the nice stuff
        #well, in the interim:

#for post in all_posts:
    #print(soup.find("eddie"))
    #NOOO this says none for each post_text

#for post in all_posts:
    #print(soup.find_all("reddie"))

###for post in all_posts:
    #print('=======')
    #print(post)
    #print('=======')
filename = 'ghost_neat.txt'
for post in all_posts
    with (filename, 'w') as fout:
        fout.write(post[0])
        fout.write('\n')
        fout.write(post[1])
        fout.write('\n=========\n')
        fout.write(post)

#counter = 0
#for post in all_post
    #with open(str(counter) + '-' + gummy_neat, 'a') as fout:
        #fout.write(the_stuff_to_write)


        ###all_posts.append((timestamp,post_text))
        #okay what happens if print all_posts?
        #test: print(all_posts)
        #oh it's terrible. okay, i get the next step. but why this one?
        #it seems to be going forever??? ack

###for post in all_posts:
    #print('=======')
    #print(post)
    #print('=======')


#okay, much harder to read. what if i make all_posts without
