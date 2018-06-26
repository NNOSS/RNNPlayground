import numpy as np
import string
import re
import sys

def getNamestxt(filename):
    '''build database from text file where it only takes first word from
    each line'''
    with open(filename,"r") as f:
        wordlist = [r.lower().replace("\n",'').split('\t')[0] for r in f]
    return wordlist

def getNamescsv(filename):
    '''build database from csv where it only takes the first word ff first column'''
    wordlist = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            f = str(row[0])
            # print(type(f))
            f = re.split("[, '\-!?:.$12368&]+", f)
            # print(f)

            if len(f[0]) > 2: wordlist.append(f[0].lower())
            # for i in f[0].lower():
            #     if i not in string.ascii_lowercase:
            #         print(i)
    return wordlists

def get_batch_generator(filename, batch_size, length):
    wordlist = getNamestxt(filename)
    i = 0
    while i + batch_size < len(wordlist):
        batch = np.zeros([length, batch_size], dtype=np.int8)
        for j, word in enumerate(wordlist[i:i+batch_size]):
            indices = [string.ascii_lowercase.index(letter) + 1 for letter in word]
            if len(indices) > length:
                indices = indices[:length]
            batch[np.arange(len(indices)), j] = indices
        yield batch

def print_words(example):
    print(example)
    for word in example[0]:
        for value in word:
            value = np.squeeze(value)
            if value != 0:
                sys.stdout.write(string.ascii_lowercase[value-1])
                sys.stdout.flush()
        print('')


if __name__ == "__main__":
    bg = get_batch_generator('malenames.txt', 10, 5)
    a = next(bg)
    print(a[0])
    print(a[:,0])
