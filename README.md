# Automatisation for  articles' classification with their description or photo

## Table des matières :
1. [Data](#data)
2. [Description](#description)
3. [Photos](#photo)

## Data

One DataFrame with 1050 articles and 14 columns  
1050 photos of the articles


## Description

From article's description, find the category.  
We will see how to process the description to extract only valuable words and predict a category

EDA :
* Explore the data
* Select column
* Extract category

Preprocessing :
* Stop words
* lemmatization and stemming
* Tag POS
* Duplicates

NLP : 
* Tf-idf
* CountVectorizer
* Word2Vec
* BERT
* USE



3 Notebooks : 
* [EDA](https://github.com/Mikael-DataAnalyst/Classification-NLP-ComputerVision/blob/master/notebook_eda.ipynb)  
 
* [Preprocessing](https://github.com/Mikael-DataAnalyst/Classification-NLP-ComputerVision/blob/master/preprocessing_nlp.ipynb) 
* [NLP](https://github.com/Mikael-DataAnalyst/Classification-NLP-ComputerVision/blob/master/notebook_nlp.ipynb)  


## Photos

* SIFT
* ORB
* ImageNet(VGG16)