# KnowledGraphEmbeddings
Obtaining Embeddings from knowledge graph in RDF format to find similarities between entities through Word2Vec


This  project  aimed  to  find  interesting  relations  between  food  products  and  based  onthose relations, suggest alternatives (recommendations) to a user given as input a specificcriterion; for example:  find healthier alternatives available to a product I like.  The project was focused on using cosine distance as similarity measure of the different food products, in  an  embedding  space  obtained  using  the  RDF2vec  approach.   The  data  set  used  inthis project was the open food facts project data set.  This data set contains informationabout the food product such as: name, quantity, brand, nutritional information, ingredients,country where is sold, category among others.

Data in zip folder contains sample from the RDF (xml) file downloaded from the open food facts project, A CSV file obtained from the open food facts project regarding frood products with nutritional ratings sold in France with ingredients from France, A pre processed CSV file with features filtered out to enhance models' performance.

Models can be trained using the obtain_embeddings file where the parameters can be tuned as desired.
Text files are embeddings in gensim word2vec format.
Create tensorflow projector file creates all metadata needed to visualize the embeddings on TensorFlow Projector.
test model computes most similar products to a product (https://world.openfoodfacts.org/product/20290948) that was duplicated in the .nt data labeled as test and see if the models are able to give maximum similarity to that test duplicate entry.
Model recomendations, gives similar products using positive examples and negative examples as references.
Included in zip is a tensorflow projector metadata to visualize the embeddings obtained using Skip-Gram with 15 dimensional vectors.


In all Files:
SG : Skip-Gram
CBOW: Common Bag Of Words
FT: Fast Text
Glove: Glove
numbers after the model: dimensions of the embeddings
