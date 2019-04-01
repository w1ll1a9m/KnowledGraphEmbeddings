
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


print("CBOW_10")
print ('<https://world.openfoodfacts.org/product/3660140914253>')
print("most similar: ")
print (model1.most_similar(positive=['<https://world.openfoodfacts.org/product/3660140914253>',  '"d"'], negative=['"e"'], topn=5))
print(".\n")

print("CBOW_15")
print ('<https://world.openfoodfacts.org/product/3660140914253>')
print("most similar: ")
print (model2.most_similar(positive=['<https://world.openfoodfacts.org/product/3660140914253>',  '"d"'], negative=['"e"'], topn=5))
print(".\n")

print("SG_10")
print ('<https://world.openfoodfacts.org/product/3660140914253>')
print("most similar: ")
print (model3.most_similar(positive=['<https://world.openfoodfacts.org/product/3660140914253>',  '"d"'], negative=['"e"'], topn=5))
print(".\n")

print("SG_15")
print ('<https://world.openfoodfacts.org/product/3660140914253>')
print("most similar: ")
print (model4.most_similar(positive=['<https://world.openfoodfacts.org/product/3660140914253>',  '"d"'], negative=['"e"'], topn=5))
print(".\n")
