import argparse
import gzip, os, csv
import numpy as np
import random
import time
import gensim, logging, os, sys, gzip
from glove import Corpus, Glove
import os
from gensim.scripts.glove2word2vec import glove2word2vec
os.environ['KMP_DUPLICATE_LIB_OK']='True'



def randomNWalkUniform(triples, n, walks, path_depth):
    path=[]
    for k in range(walks):
        walk = randomWalkUniform(triples, n, path_depth)
        path.append(walk)
    return path

def addTriple(net, source, target, edge):
    if source in net:
        if  target in net[source]:
            net[source][target].add(edge)
        else:
            net[source][target]= set([edge])
    else:
        net[source]={}
        net[source][target] =set([edge])
            
def getLinks(net, source):
    if source not in net:
        return {}
    return net[source]

# Generate paths (entity->relation->entity) by radom walks
def randomWalkUniform(triples, startNode, max_depth=4):
    next_node =startNode
    path = str(startNode)+'->'
    for i in range(max_depth):
        neighs = getLinks(triples,next_node)
        #print (neighs)
        if len(neighs) == 0: break
        weights = []
        queue = []
        for neigh in neighs:
            for edge in neighs[neigh]:
                queue.append((edge,neigh))
        edge, next_node = random.choice(queue)
        path = path +str(edge)+'->'
        path = path +str(next_node)+'->'
    path =path.split('->')
    return path


def preprocess(fname):
    triples = {}

    ent_counter = 0
    rel_counter = 0
    train_counter = 0

    print (fname)
    #gzfile= gzip.open(fname, mode='rt')

    for line in gzip.open(fname, mode='rt'):
        #print (line)
        line = line.rstrip('\n.')
        words = line.split(" ")
        h = words[0]
        r = words[1]
        t = words[2]
        
        train_counter +=1

        addTriple(triples, h, t, r)
        train_counter+=1
    print ('Triple:',train_counter)
    return triples


file = 'file in .gz format'
triples = preprocess(file)
entities = list(triples.keys())
vocabulary = entities
print (len(vocabulary))

walks = 80
path_depth = 5
paths = randomNWalkUniform(triples, entities[0], walks, path_depth)
print(paths)



start_time =time.time()
sentences =[]

for word in vocabulary:
    sentences.extend( randomNWalkUniform(triples, word, walks, path_depth) )

elapsed_time = time.time() - start_time
print ('Time elapsed to generate features:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#de models

 #GloVe
corpus = Corpus()
corpus.fit(sentences, window=10)
glove_500 = Glove(no_components=10, learning_rate=0.05)
glove_500.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove_500.add_dictionary(corpus.dictionary)
glove_500.save('glove_10.model')

#GloVe 
glove_200 = Glove(no_components=15, learning_rate=0.05)
glove_200.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove_500.add_dictionary(corpus.dictionary)
glove_500.save('glove_15.model')



#fasttext 500
print("start fast 10")
modelf = gensim.models.FastText(size=10, workers=5, window=10, sg=1, negative=15, iter=30)
modelf.build_vocab(sentences)
#print("vocabulary: ",len(model.wv.vocab))
total_examples=modelf.corpus_count
modelf.train(sentences,total_examples,epochs=30)
#sg/cbow features iterations window negative hops random walks
modelf.save('FT_10')
vectorsf = modelf.wv
vectorsf.save_word2vec_format("FT_10.txt",binary = False)

del modelf
#fasttext 200 
print("start fast 15")
modelf2 = gensim.models.FastText(size=15, workers=5, window=10, sg=1, negative=15, iter=30)
modelf2.build_vocab(sentences)
total_examples=modelf2.corpus_count
modelf2.train(sentences,total_examples,epochs=30)
#sg/cbow features iterations window negative hops random walks
modelf2.save('FT_15')
vectorsf2 = modelf2.wv
vectorsf2.save_word2vec_format("FT_15.txt",binary = False) 

del modelf2 

 #sg Skip-Gram
print("start sg 15")
model = gensim.models.Word2Vec(size=5, workers=5, window=10, sg=1, negative=15, iter=30)
model.build_vocab(sentences)
total_examples=model.corpus_count
print("vocabulary: ",len(model.wv.vocab))
model.train(sentences,total_examples,epochs=30)
#sg/cbow features iterations window negative hops random walks
model.save('SG_15')
vectors = model.wv
vectors.save_word2vec_format("SG_15.txt",binary = False)


#sg Skip-Gram
model1 = gensim.models.Word2Vec(size=3, workers=5, window=5, sg=1, negative=15, iter=30)
model1.reset_from(model)


#sg Skip-Gram
model2 = gensim.models.Word2Vec(size=3, workers=5, window=5, sg=0, iter=30,cbow_mean=1, alpha = 0.05)
model2.reset_from(model)


#sg Skip-Gram
model3 = gensim.models.Word2Vec(size=5, workers=5, window=5, sg=0, iter=30, cbow_mean=1, alpha = 0.05)
model3.reset_from(model)

del model
print("start sg 10")
model1.train(sentences,total_examples,epochs=30)
model1.save('SG_10')
vectors1 = model1.wv
vectors1.save_word2vec_format("SG_10.txt",binary = False)

del model1
print("start cbow 10")
model2.train(sentences,total_examples,epochs=30)
model2.save('CBOW_10')
vectors2 = model2.wv
vectors2.save_word2vec_format("CBOW_10.txt",binary = False)


del model2
print("start cbow 15")
model3.train(sentences,total_examples,epochs=30)
model3.save('CBOW_15')
vectors3 = model3.wv
vectors3.save_word2vec_format("CBOW_15.txt",binary = False)
