# Twitter sentiment analysis

Project 2 of EPFL Machine Learning course: Twitter sentiment analysis

**Authors**: Rayan Daod Nathoo, Yann Meier, Kopiga Rasiah.

**Deadline**: 19.12.2019

The goal of this sentiment analysis project is to classify whether a tweet is positve or negative by considering its text only. We have used Python 3.6 to implement this project.


## Getting started

To start, please clone this repository.

In order to run our project, you will need to install the following modules:

`Numpy`

`Keras with TensorFlow backend`

`TextBlob`

`NLTK`

`Wordsegment`

`Autocorrect`


Please use usual command as `pip install [module]` by changing [module] to the corresponding module name.

### Folder structures.

Create a folder `data`in the repository at the root of the project. 
Inside `data`, create the folders `preprocessed` and `glove.twitter.27B`.
Again inside `data`, place the training and test sets retrieved from AICrowd.


In `preprocessed`, create the empty folders `neg`, `pos`, `test`.

Download the file http://nlp.stanford.edu/data/glove.twitter.27B.zip and place it inside the folder `glove.twitter.27B`. At the end, you should obtain the following folder structure:


#### Folder structure
------------

    ├── Tweet-classification                     
        ├── data
            ├── glove.twitter.27B 
                ├── glove.twitter.27B.200d.txt      The file regrouping all the pre-trained embedding vectors we used for our                                                    algorithm.
            ├── preprocessed       
            ├── pos
            ├── neg
            ├── test
        ├── src
            ├── embeddings
                ├── __init__.py
                ├── cooc.py                         Generates a coocurrence matrix from the words of our vocabulary.
                ├── glove_GD.py                     Implements a Gradient Descent version of Glove.
                ├── stanford_word_embedding.py      Creates a vocabulary based on Twitter pre-trained Stanford Glove vectors.
                ├── tf_idf.py                       Regroups some functions we used for an alternative method.
                ├── tweet_embeddings.py             Creates tweet embeddings from the word embeddings.
                ├── vocab.py                        Takes care about creating a vocabulary from our corpus.
            ├── prediction
                ├── __init__.py
                ├── better_predict.py               Regroups all the implementations we tried for the training part.
                ├── predict.py                      Stores the two training algorithms we used in the end.
            ├── preprocessing
                ├── __init__.py
                ├── dictionaries.py                 Regroups the dictionaries we used during the preprocessing part.
                ├── preprocess.py                   Regroups all the preprocessing algorithms we implemented.
           ├── __init__.py
           ├── params.py                            Regroups all the parameters that control this project.
           ├── paths.py                             Regroups all the file paths required for our algorithm.
           ├── run.py                               To be run after the above instructions to execute our pipeline.
        ├── .gitignore

--------

### Technical Overview

1. Preprocessing:
- Remove the tweets that are in both positive and negative tweet files
- Process the spaces in the training set of tweets
- Expand the english contractions contained in the training set
- Separate words from hashtags

2. Embeddings:
- The word embeddings were taken in https://nlp.stanford.edu/projects/glove/.
- The tweet embeddings are the sum of the word embeddings of the words it contains.

3. Prediction:
- We used different models to predict. The best one was a Neural Network with one hidden layer of 256 nodes (c.f report for more informations).

