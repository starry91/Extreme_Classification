import bcolz
import pickle
import numpy as np  
import csv

glovePath = "glove"

def constructWeightMatrix(filename, datafile, embd_dim):
    words = []
    idx = 0
    word2idx = {}
    glove_path = "."
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.{embd_dim}.dat', mode='w')

    with open(f'{glove_path}/glove.6B.{embd_dim}d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
        
    vectors = bcolz.carray(vectors[1:].reshape((-1, embd_dim)), rootdir=f'{glove_path}/6B.{embd_dim}.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.{embd_dim}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.{embd_dim}_idx.pkl', 'wb'))
    vectors = bcolz.open(f'{glove_path}/6B.{embd_dim}.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.{embd_dim}_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.{embd_dim}_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}


    with open(datafile, newline='') as csvfile:
        words = list(csv.reader(csvfile))
    matrix_len = len(words)
    weights_matrix = np.zeros((matrix_len, embd_dim))
    words_found = 0

    for i, word in enumerate(words):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(embd_dim, ))

    np.savetxt(filename, weights_matrix, delimiter = ",")