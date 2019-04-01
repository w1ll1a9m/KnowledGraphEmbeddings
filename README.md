# KnowledGraphEmbeddings
Obtaining Embeddings from knowledge graph in RDF format to find similarities between entities through Word2Vec


This  project  aimed  to  find  interesting  relations  between  food  products  and  based  onthose relations, suggest alternatives (recommendations) to a user given as input a specificcriterion; for example:  find healthier alternatives available to a product I like.  The projectwas focused on using cosine distance as similarity measure of the different food products,in  an  embedding  space  obtained  using  the  RDF2vec  approach  [1].   The  data  set  used  inthis project was the open food facts project data set.  This data set contains informationabout the food product such as: name, quantity, brand, nutritional information, ingredients,country where is sold, category among others.


Text files are embeddings in gensim word2vec format.
Models can be trained using the obtain_embeddings file where the parameters can be tuned as desired.


In all Files:
SG : Skip-Gram
CBOW: Common Bag Of Words
FT: Fast Text
Glove: Glove
numbers after the model: dimensions of the embeddings
