'''
Created on Feb 16, 2016

@author: petar
'''
import gensim, logging, os, sys
from glove import Corpus, Glove





model1 = gensim.models.Word2Vec.load('CBOW_10')
model2 = gensim.models.Word2Vec.load('CBOW_15')
model3 = gensim.models.Word2Vec.load('SG_10')
model4 = gensim.models.Word2Vec.load('SG_15')
model5 = gensim.models.Word2Vec.load('FT_10')
model6 = gensim.models.Word2Vec.load('FT_15')
model7 = Glove.load('glove_10.model')
model8 = Glove.load('glove_15.model')

#product 3350030209940 duplicated in original rdf tripples file


print("CBOW_10")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model1.most_similar(positive=['<https://world.openfoodfacts.org/product/3350030209940>'], topn=5))
print(".\n")

print("CBOW_15")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model2.most_similar(positive=['<https://world.openfoodfacts.org/product/3350030209940>'], topn=5))
print(".\n")

print("SG_10")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model3.most_similar(positive=['<https://world.openfoodfacts.org/product/3350030209940>'], topn=5))
print(".\n")

print("SG_15")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model4.most_similar(positive=['<https://world.openfoodfacts.org/product/3350030209940>'], topn=5))
print(".\n")

print("FT_10")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model5.most_similar(positive=['<https://world.openfoodfacts.org/product/3350030209940>'], topn=5))
print(".\n")

print("FT_15")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model6.most_similar(positive=['<https://world.openfoodfacts.org/product/3350030209940>'], topn=5))
print(".\n")

print("Glove_10")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model7.most_similar('<https://world.openfoodfacts.org/product/3350030209940>', number=5))
print(".\n")

print("Glove_15")
print ('<https://world.openfoodfacts.org/product/3350030209940>')
print("most similar: ")
print (model8.most_similar('<https://world.openfoodfacts.org/product/3350030209940>', number=5))
print(".\n")
