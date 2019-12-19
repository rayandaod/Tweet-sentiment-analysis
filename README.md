# Tweet-classification
Project 2 of EPFL Machine Learning course: tweet classification. 
Authors: Rayan Daod Nathoo, Yann Meier, Kopiga Rasiah.
Deadline: 19.12.2019.
This is our project implements the twitter sentiment analysis challenge. The goal of this project is to classify whether a tweet is positve or negative by considering it's text only. We have used python 3.6 to implement this project.



## Getting started

### Clone the repository.


### Required packages
In order to run our project, you will need to install the following modules: 

`Numpy`

`Keras with TensorFlow backend`

`TextBlob`

`NLTK`

`Wordsegment`

`Autocorrect`


Please use usual command as `pip install [module]` by changing [module] to the corresponding module name.

### Folder structures.

Create a folder `data`in the repository at the root of the project. Inside `data`, create the folders `preprocessed` and `glove.twitter.27B`
In `preprocessed`, create the folders `neg`, `pos`, `test`.
Download the file http://nlp.stanford.edu/data/glove.twitter.27B.zip in `glove.twitter.27B`. At the end, you should have the following folder structure.


#### Folder structure
------------

    ├── Tweet-classification                     
        ├── data
            ├── glove.twitter.27B 
                ├── glove.twitter.27B.200d.txt
            ├── preprocessed       
            ├── pos
            ├── neg
            ├── test
        ├── src
            ├── embeddings
                ├── __init__.py
                ├── als.py
                ├── cooc.py
                ├── glove_solution.py
                ├── stanford_word_embedding.py
                ├── tf_idf.py
                ├── tweet_embeddings.py
                ├── vocab.py
            ├── prediction
                ├── __init__.py
                ├── better_predict.py
                ├── predict.py
            ├── preprocessing
                ├── __init__.py
                ├── dictionaries.py
                ├── preprocess.py
           ├── __init__.py
           ├── params.py                            : Control tower, see below for more details.
           ├── paths.py 
           ├── run.sh
        ├── Useful references
        ├── .gitignore
        ├── project2_description.pdf
        ├── Project_Guidlines.pdf

--------

#### Technical Overview

1. Preprocessing. 
- remove tweets that are in both positive tweet file and negativ tweet file
- process spaces
- process contractions in english
- segment expressions coming with hashtags.

2. Embeddings.
- The word embeddings are given by https://nlp.stanford.edu/projects/glove/. 

3. Predict.
- We used different models to predict. The best one was CNN with tweet embeddings and we got the following 

  

