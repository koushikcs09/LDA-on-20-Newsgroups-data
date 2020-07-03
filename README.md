# LDA-on-20-Newsgroups-data
##	LDA (Latent Dirichlet Allocation) Clustering
LDA is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions.

-   Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
-   LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial.
-   It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution.

## Load the packages
The core package used in this tutorial is scikit-learn (`sklearn`) and Gensim(`gensim`).

Regular expressions  `re`,  `gensim`  and  `spacy`  are used to process texts.  `pyLDAvis`  and  `matplotlib`  for visualization and  `numpy`  and  `pandas`  for manipulating and viewing data in tabular format.
```python
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
%matplotlib inline

import pickle # for saving and loading objects
#from gensim.models import CohermodelsenceModel
import gensim.corpora as corpora
from nltk.tag import pos_tag
from gensim.utils import simple_preprocess
import os
import time
import datetime
import tarfile
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from collections import OrderedDict
```
## Import Newsgroups Text Data
Dataset contains about 11k newsgroups posts from 20 different topics.
```python
from sklearn.datasets import fetch_20newsgroups  
newsgroups_train = fetch_20newsgroups(subset=’train’, shuffle = True)  
newsgroups_test = fetch_20newsgroups(subset=’test’, shuffle = True)
print(newsgroups_train.head())
```
![Input - 20Newsgroups](https://www.machinelearningplus.com/wp-content/uploads/2018/04/input_texts-1024x608.png?ezimgfmt=rs:722x428/rscb1/ng:webp/ngcb1)
This data set has the news already grouped into key topics. Which you can get by
```python
print(list(newsgroups_train.target_names))
```
There are 20 targets in the data set —
‘alt.atheism’,  
‘comp.graphics’,  
‘comp.os.ms-windows.misc’,  
‘comp.sys.ibm.pc.hardware’,  
‘comp.sys.mac.hardware’,  
‘comp.windows.x’,  
‘misc.forsale’,  
‘rec.autos’,  
‘rec.motorcycles’,  
‘rec.sport.baseball’,  
‘rec.sport.hockey’,  
‘sci.crypt’,  
‘sci.electronics’,  
‘sci.med’,  
‘sci.space’,  
‘soc.religion.christian’,  
‘talk.politics.guns’,  
‘talk.politics.mideast’,  
‘talk.politics.misc’,  
‘talk.religion.misc

Looking visually we can say that this data set has a few broad topics like:

-   Science
-   Politics
-   Sports
-   Religion
-   Technology etc
## Remove emails and newline characters

You can see many emails, newline characters and extra spaces in the text and it is quite distracting. Let’s get rid of them using [regular expressions].
```python
# Remove Emails
texts = [re.sub('\S*@\S*\s?', '', sent) for sent in texts]

# Remove new line characters
texts = [re.sub('\s+', ' ', sent) for sent in texts]

# Remove distracting single quotes
texts = [re.sub("\'", "", sent) for sent in texts]

pprint(texts[:1])
```
```
['From: (wheres my thing) Subject: WHAT car is this!? Nntp-Posting-Host: '
 'rac3.wam.umd.edu Organization: University of Maryland, College Park Lines: '
 '15 I was wondering if anyone out there could enlighten me on this car I saw '
 'the other day. It was a 2-door sports car, looked to be from the late 60s/ '
 'early 70s. It was called a Bricklin. The doors were really small. In '
 'addition, the front bumper was separate from the rest of the body. This is '
 'all I know. If anyone can tellme a model name, engine specs, years of '
 'production, where this car is made, history, or whatever info you have on '
 'this funky looking car, please e-mail. Thanks, - IL ---- brought to you by '
 'your neighborhood Lerxst ---- ']
```

## Pre-Processing : Cleaning, Tokenization and Lemmatization
The sentences look better now, but you want to tokenize each sentence into a list of words, removing punctuation and unnecessary characters altogether.
- **Omitting Text have lesser words:** The Text needs to be meaningful and should be having minimum number of words to proceed with the analysis. The Text entries having lesser words than the user specified threshold are removed and the remaining text entries are considered for further analysis.
```python
count=texts.str.split().str.len()
df_cln=df[(count>n)]
```
- **Auto-correction of spelling:** The Text might be having incorrect word spellings and it needs to be corrected before proceeding with the text analysis.
```python
texts=TextBlob(texts).correct()
```
- **Remove Unicode:** The Unicode characters <\\u[0-9A-Fa-f]+ ^\x00-\x7f> present in the incoming text needs to be removed.
```python
texts= re.sub(r'(\\u[0-9A-Fa-f]+)',r' ', texts)       
texts= re.sub(r'[^\x00-\x7f]',r' ',texts)
```
-  **Replace URL:** The URL or web link characters < ((www\.[^\s]+)|(https?://[^\s]+))> present in the string needs to be cleansed.
```python
texts= re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',texts)
```
- **Replace @User:** The text file might have contains <@user>, which needs to be removed before processing the file.
 ```python
 texts= re.sub(r'#([^\s]+)', r'\1', texts)
 ```
-	**Remove Hash Tag:** The Hash Tag Characters <#> present in the text needs to be removed.
```python
texts= re.sub(r'#([^\s]+)', r'\1', texts)
```
-	**Remove numbers, special characters, multiple exclamation mark, multiple full stop and multiple commas.**
```python
texts= ''.join([i for i in textsif not i.isdigit()])
texts= re.sub(r"(\!)\1+", ' ', texts)
texts= re.sub(r"(\?)\1+", ' ', texts)
texts= re.sub(r"(\.)\1+", '.', texts)
texts= ' '.join(re.sub("(@[A-Za-z[0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\/S+)", " ", texts).split())
```
- **Replace incomplete words:** The incomplete words /abbreviations are identified within the text and replaced with the correct words.
```python
 texts=texts.replace(“ci”,“called in”)
 ```
-	**Remove System Words:** The customized system words dictionary needs to be created based on the context of the data and needs to be removed from the text.
-	**Remove names:** The names present in the text field are redundant in our analysis to determine the topic of the text and hence excluded from the text using python spacy package.
```python
doc=nlp(texts)
      for ent in doc.ents:
          texts= texts.replace(ent.text,"")
```
- **Remove Stopwords:** The most commonly used English words and prepositions are removed from the text (using nltk.corpus stopwords) to get more meaningful insight from the data.
```python
Stopwords_list=stopwords.words('english')
```
- **Tokenization:** The Sentence has been split into individual tokens using the python NLTK word tokenize. 
```python
tokens = nltk.word_tokenize(texts)
```
- **Lemmatization**
- Once the Text cleansing is done in above steps, the lemma (root word) has been identified from the individual tokens using NLTK wordnet lemmatizer based on the significance of the word (NLTK pos-tag) in the corresponding sentence.
- 
For example: ‘Studying’ becomes ‘Study’, ‘Meeting becomes ‘Meet’, ‘Better’ and ‘Best’ becomes ‘Good’.

The advantage of this is, we get to reduce the total number of unique words in the dictionary. As a result, the number of columns in the document-word matrix (created by CountVectorizer in the next step) will be denser with lesser columns.
```python
lemma = WordNetLemmatizer()
Noun tag (pos=n) => 		['NN', 'NNS', 'NNP', 'NNPS']
Verb tag (pos=v)=>		['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
Adjective tag (pos=a) =>	['JJ', 'JJR', 'JJS']
Adverb tag (pos=r)=> 	['RB', 'RBR', 'RBS']
Else (pos=n)
lemma_word = lemma.lemmatize(word=word, pos=pos)
```
```python
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
texts_mod = lemmatization(texts_mod, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(texts_mod[:2])
```
```
['where s  thing subject what car be nntp post host rac wam umd edu organization university maryland college park line be wonder anyone out there could enlighten car see other days be door sport car look be late early be call bricklin door be really small addition front bumper be separate rest body be know anyone can tellme model name engine spec year production where car be make history whatev info have funky look car mail thank bring  neighborhood lerxst' (..truncated..)]
```
 - **Preprocessing:** The Python package Gensim provides a preprocessing function to preprocess the data and make it ready for clustering.
```python
gensim.utils.simple_preprocess(text),deacc=True)
```
### Standard Text Cleansing Steps
```python
# Data Cleansing
for text1 in texts:
    text = text1
    #remove_names()
    new_sentence = []
    text_split=text.split(" ")
    tagged_sentence = pos_tag([word for word in text_split if word])
    for word, tag in tagged_sentence:
        if tag in ['NNP', 'NNPS']:
            lemma_word = ""
        else:
            lemma_word = word

        new_sentence.append(lemma_word)
    text2=""
    for i in new_sentence:
        text2 = text2 + " " + i
    text=text2

    # Converting to Lower case
    text=text.lower()

    # Removing unwanted information 
    text=text.replace("-"," ").replace(".com"," ")
    text=' '.join(re.sub("(@[A-Za-z[0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\/S+)", " ", text).split())

    # Adding Space at begining and End of the Text
    text=" " + text + " "

    #smart_lemmatize()
    new_sentence = []
    lemma = WordNetLemmatizer()
    text_split=text.split(" ")
    tagged_sentence = pos_tag([word for word in text_split if word])
    for word, tag in tagged_sentence:
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            pos = 'n'
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            pos = 'v'
        elif tag in ['JJ', 'JJR', 'JJS']:
            pos = 'a'
        elif tag in ['RB', 'RBR', 'RBS']:
            pos = 'r'
        else:
            pos = 'n'
        lemma_word = lemma.lemmatize(word=word, pos=pos)
        new_sentence.append(lemma_word)
    text2=""
    for i in new_sentence:
        text2 = text2 + " " + i
    text=text2
```
```python
	#removeUnicode()
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r' ', text)       
    text = re.sub(r'[^\x00-\x7f]',r' ',text)

    #replaceURL()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)

    #replaceAtUser()
    text = re.sub('@[^\s]+',' ',text)

    #removeHashtagInFrontOfWord()
    text = re.sub(r'#([^\s]+)', r'\1', text)

    #removeNumbers()
    text = ''.join([i for i in text if not i.isdigit()]) 

    #replaceMultiExclamationMark()
    text = re.sub(r"(\!)\1+", ' ', text)

    #replaceMultiQuestionMark()
    text = re.sub(r"(\?)\1+", ' ', text)

    #replaceMultiStopMark()
    text = re.sub(r"(\.)\1+", '.', text)

    #replace_incomplete_word()
    #remove_stop_words()
    from nltk.corpus import stopwords
    rmv_wrd_lst=stopwords.words('english')
    rmv_stop_word=[]
    for i in rmv_wrd_lst:
        #print(i)
        wrd=" " + str(i) + " "
        rmv_stop_word.append(wrd)
    for t in rmv_stop_word:
        text=text.replace(t," ")

    #replaceMultiSpace()
    #text=text.replace("  "," ")
    text = re.sub(r"(\ )\1+", ' ', text)        

    text_corpus.append(gensim.utils.simple_preprocess(str((text.split()))))
    #texts.append(gensim.utils.simple_preprocess(str((clean_text_ngram(text1)).split()),deacc=True))
    texts_mod.append(text)
    dictionary=corpora.Dictionary(text_corpus)  corpus=[dictionary.doc2bow(text)  for  text  in  text_corpus]  coherence_values  =  []  model_list  =  []
```
## Create the Document-Word matrix
The LDA topic model algorithm requires a document word matrix as the main input.
Here we compare LDA results for below 3 types of such conversion methods.

 - Bag of words(BoW)
  - TfidfVectorizer
  - CountVectorizer

### 1. Bag of words(BoW)
#### Converting text to bag of words
**Bag of words(BoW)** like a dictionary where the key is the word and value is the number of times that word occurs in the entire corpus.
```python
import  gensim
from  gensim.corpora.dictionary  import  Dictionary
dictionary=corpora.Dictionary(text_corpus)
text_corpus=[]
text_corpus.append(gensim.utils.simple_preprocess(str((text.split()))))
dictionary = gensim.corpora.Dictionary(processed_docs)
```
Now for each pre-processed document we use the dictionary object just created to convert that document into a bag of words. i.e for each document we create a dictionary reporting how many words and how many times those words appear.
```python
corpus=[dictionary.doc2bow(text)  for  text  in  text_corpus]
```
The results look like:
```
Word 453 ("exampl") appears 1 time.  
Word 476 ("jew") appears 1 time.  
Word 480 ("lead") appears 1 time.  
Word 482 ("littl") appears 3 time.  
Word 520 ("wors") appears 2 time.  
Word 721 ("keith") appears 3 time.  
Word 732 ("punish") appears 1 time.  
Word 803 ("california") appears 1 time.  
Word 859 ("institut") appears 1 time.
```
**Running LDA**
This is actually quite simple as we can use the gensim LDA model. We need to specify how many topics are there in the data set. Lets say we start with 8 unique topics. Num of passes is the number of training passes over the document.
##### Hyper parameter tuning (n_topics,$\alpha$(doc_topic_prior),$\beta$(topic_word_prior), learning decay)
```python
coherence_values = []
model_list = []

limit=13
start=10
step=1
topics_range = range(start, limit, step)
with open('Gensim LDA (Hypereparameter Tuning)_BoW.txt', 'a', encoding='utf-8') as file:
    file.write('Gensim LDA (Hypereparameter Tuning) [Bag of Words]!')
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

model_results = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }
# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)))
    

    # iterate through number of topics
    for k in topics_range:
        # iterate through alpha values
        for a in alpha:
            # iterare through beta values
            for b in beta:
                # get the coherence score for the given parameters
                model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
            
                coherencemodel = CoherenceModel(model=model,texts=text_corpus,corpus=corpus,dictionary=dictionary, coherence='c_v')
                print(' Topics ' , k)
                print(' Alpha ' , a)
                print(' Beta ' , b)
                print(' CV ' , coherencemodel.get_coherence())
                coherence_values.append(coherencemodel.get_coherence())
                model_list.append(k)
                
                # Save the model results 
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(coherencemodel.get_coherence())
                with open('Gensim LDA (Hypereparameter Tuning)_BoW.txt', 'a', encoding='utf-8') as file:
                    file.write(model_results)
                pbar.update(1)
    pd.DataFrame(model_results).to_csv('Gensim LDA (Hypereparameter Tuning) [BoW]lda_tuning_results.csv', index=False)
    pbar.close()
```
####	Visualize the LDA Clusters
The LDA Clusters can be visualized with top keywords and prevalence using the pyLDAvis gensim package.
```python
:

# Final LDA Model with optimized number of clusters as identified in the above coherence plot
lda_model =  gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=10,update_every=1,chunksize=10000,passes=1,random_state=100,
                                             alpha='auto', eta='auto', decay=0.9, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, ns_conf=None, minimum_phi_value=0.01, per_word_topics=False, callbacks=None)
# Saving the Final LDA Model and Data as Pickle file
with open ('BOW_text_corpus_20Newgroup_' + dt_stamp + 'pkl', 'wb') as pkl_fl:
        pickle.dump(text_corpus,pkl_fl)        
with open ('BOW_lda_model_20Newgroup_' + dt_stamp + 'pkl', 'wb') as pkl_fl:
        pickle.dump(lda_model,pkl_fl)
with open ('BOW_data_vectorized_20Newgroup_' + dt_stamp + 'pkl', 'wb') as pkl_fl:
        pickle.dump(data_vectorized,pkl_fl)        
# LDA Visualization
pyLDAvis.enable_notebook()
viz=pyLDAvis.gensim.prepare(lda_model,corpus,dictionary,sort_topics=False)
# save the LDA visualization
pyLDAvis.save_html(viz,'BOW_lda_visualization_20Newgroup_' + dt_stamp + '.html')
pyLDAvis.display(viz)
```
#### LDA Output
The LDA output has been extracted and feed to Tableau for better visualizations and word-cloud.
```python
lda_model.print_topics(o)

1st output:
cluster_view.topic_info 
<’document’,’word’,’freq’,’total’>

2nd Output:
lda_model.per_word_topics
<'Document_No','Dominant_Topic','Topic_Perc_Contrib','keywords','Text'>
```
```python
# LDA Output
freq_df = viz.topic_info
freq_df['prevalence'] = freq_df['Freq'] / freq_df['Total'] # calculates prevalence ratio (0 – 1)
freq_df = freq_df.loc[freq_df['Category'] != 'Default'] # filters out extra data
freq_df.to_excel('BOW_20Newgroup_keywords_' + dt_stamp + '.xlsx') # exports to excel
```
#### LDA keywords to Topic
The n-gram (bigram, trigram) combinations have been generated from the keywords identified in LDA clustering. The highest occurring n-gram combination from each of the cluster has been chosen in the nomenclature of the corresponding cluster.
For e.g : Suppose a cluster is having following bigram combination sorted in descending order, the name of the cluster has been automatically chosen as top bigram combination “clear weather”.  
clear weather -> 50 times
Rainy weather -> 10 times 
```python
sent_topics_df = pd.DataFrame()
for i,row_list in enumerate(lda_model[corpus]):
    row = row_list[0] if lda_model.per_word_topics else row_list  # get list of sentences
    row = sorted(row, key=lambda x: (x[1]), reverse=True)

    for j , (topic_num,prop_topic) in enumerate(row):
        if j == 0:
            wp =lda_model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word,prop in wp])
            sent_topics_df=sent_topics_df.append(pd.Series([int(topic_num),round(prop_topic,4),topic_keywords]),ignore_index=True)
        else:
            break
sent_topics_df.columns = ['Dominant_Topic','Perc_Contribution','Topic_Keywords']
contents = pd.Series(texts_mod)
contents_act = pd.Series(texts)
gensim_lda_eta = pd.Series(lda_model.eta)
sent_topics_df = pd.concat([sent_topics_df,contents_act,contents,gensim_lda_eta],axis =1)

df_dominant_topic = sent_topics_df.reset_index()
df_dominant_topic.columns = ['Document_No' ,'Dominant_Topic','Topic_Perc_Contrib','keywords','Original Text','Clean Text','eta/beta']
lda_alpha = pd.Series(lda_model.alpha)
lda_topic = pd.Series(range(0,12))
comb_topic_alpha = pd.concat([lda_topic,lda_alpha],axis =1)
comb_topic_alpha.columns = ['Dominant_Topic','alpha/theta']
df_dominant_topic  = pd.merge(df_dominant_topic,comb_topic_alpha, on = 'Dominant_Topic', how = 'left')
df_dominant_topic.to_excel('BOW_20Newgroup_topic_by_document_' + dt_stamp + '.xlsx')
``` 
#### Dynamic finding the Matched Key from cluster keywords
The LDA output “keywords” column provides the keywords for individual clusters. Each document (text) might be having one or multiple keywords from the cluster keywords .All lemmas of each text has been compared with their corresponding LDA assigned keywords to get the matching keyword list. This matching keyword list along with other details will be feed to Tableau to generate more meaningful insight from the text.


### 2.CountVectorizer
#### Word Counts with CountVectorizer (sklearn)
The [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
You can use it as follows:

1.  Create an instance of the  _CountVectorizer_  class.
2.  Call the  _fit()_  function in order to learn a vocabulary from one or more documents.
3.  Call the  _transform()_  function on one or more documents as needed to encode each as a vector.

An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.

Because these vectors will contain a lot of zeros, we call them sparse. Python provides an efficient way of handling sparse vectors in the  [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)  package.

The vectors returned from a call to transform() will be sparse vectors, and you can transform them back to numpy arrays to look and better understand what is going on by calling the toarray() function.

If you want to materialize it in a 2D array format, call the `todense()` method of the sparse matrix like its done in the next step.
```python
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora.dictionary import Dictionary
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(texts_mod)
```
**Check the Sparsicity**
Sparsicity is nothing but the percentage of non-zero datapoints in the document-word matrix, that is  `data_vectorized`.

Since most cells in this matrix will be zero, I am interested in knowing what percentage of cells contain non-zero values.
```python
# Materialize the sparse data
data_dense = data_vectorized.todense()
# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
```
```
Sparsicity:  0.775887569365 %
```
**Running LDA**
```python
# transform sparse matrix into gensim corpus
corpus_vect_gensim = gensim.matutils.Sparse2Corpus(data_vectorized, documents_columns=False)
dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
```
##### Hyper parameter tuning (n_topics,$\alpha$(doc_topic_prior),$\beta$(topic_word_prior), learning decay)
```python
coherence_values = []
model_list = []
limit=13
start=10
step=1
topics_range = range(start, limit, step)
with open('Gensim LDA (Hypereparameter Tuning)_CountVectorizer.txt', 'a', encoding='utf-8') as file:
    file.write('Gensim LDA (Hypereparameter Tuning) [CountVectorizer]!')
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    model_results = {'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                    }
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)))


        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    model = gensim.models.LdaMulticore(corpus=corpus_vect_gensim,
                                               id2word=dictionary,
                                               num_topics=k, 
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               alpha=a,
                                               eta=b)

                    coherencemodel = CoherenceModel(model=model, texts=text_corpus,corpus=corpus_vect_gensim,dictionary=dictionary, coherence='c_v')
                    print(' Topics ' , k)
                    print(' Alpha ' , a)
                    print(' Beta ' , b)
                    print(' CV ' , coherencemodel.get_coherence())
                    coherence_values.append(coherencemodel.get_coherence())
                    model_list.append(k)

                    # Save the model results                    
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(coherencemodel.get_coherence())                
                    file.write(str(k)+str('-----')+str(a)+str('-----')+str(b)+str('-----')+str(coherencemodel.get_coherence())+str('-----'))
                    pbar.update(1)
        pd.DataFrame(model_results).to_csv('Gensim LDA (Hypereparameter Tuning) [CountVectorizer]lda_tuning_results.csv', index=False)
        pbar.close()
```
#### CountVectorizer (gensim )
```python
data_vectorized = vectorizer.fit_transform(texts_mod)
# transform sparse matrix into gensim corpus
corpus_vect_gensim = gensim.matutils.Sparse2Corpus(data_vectorized, documents_columns=False)
dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
coherence_values = []
model_list = []
i_model_list = []
i_cv_list = []
limit=21
start=2
step=1
for num_topics in range(start, limit, step):
            print('Number of Topics', num_topics)            
            model = gensim.models.LdaModel(corpus_vect_gensim,num_topics=num_topics, id2word=dictionary,random_state=100,chunksize=10000,passes=2,alpha='auto',eta='auto')
            
            coherencemodel = CoherenceModel(model=model, texts=text_corpus,corpus=corpus_vect_gensim,dictionary=dictionary, coherence='c_v')
            print(' CV ' , coherencemodel.get_coherence())
            coherence_values.append(coherencemodel.get_coherence())
            model_list.append(num_topics)
    
limit=21
start=2
step=1           
get_ipython().run_line_magic('matplotlib', 'inline')
x=range(start,limit,step)
plt.plot(x,coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence Score")
plt.legend("Coherence_values",loc='best')
plt.grid()
work_dir=os.getcwd()
title_name="COUNTVECTOR_GENSIM_20_newsgroup_elbow_plot"
img_name=str(work_dir)+'/'+str(title_name)+str("_")+str(dt_stamp)+ str(".png")
#fig=plt.figure(figsize=(6,4))
plt.savefig(img_name,dpi=300,bbox_inches='tight')
plt.show()
plt.clf()
```

###  Tfidf Vectorizer (Term Frequency-Inverse Data Frequency)
**TF-IDF** stands for  **“Term Frequency — Inverse Document Frequency”**. This is a technique to calculate the weight of each word signifies the importance of the word in the document and corpus. This algorithm is mostly using for the retrieval of information and text mining field.

## Term Frequency (TF)

The number of times a word appears in a document divided by the total number of words in the document. Every document has its term frequency.

![](https://miro.medium.com/max/343/0*0Uzik-cTMA-i6BUt.png)

## Inverse Data Frequency (IDF)

The log of the number of documents divided by the number of documents that contain the word  **_w_**. Inverse data frequency determines the weight of rare words across all documents in the corpus.

![](https://miro.medium.com/max/390/0*t2Uxb_43L3vjwDPm.png)

Lastly, the  **TF-IDF**  is simply the TF multiplied by IDF.

**TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)**
![](https://miro.medium.com/max/505/0*yJm1bH6Ds0vFFyhP.png)
#### Generated TF-IDF by using TfidfVectorizer from (Sklearn)
```python
vectorizer = TfidfVectorizer(analyzer='word',stop_words=rmv_stop_word)
data_vectorized = vectorizer.fit_transform([' '.join(x) for x in text_corpus])
# Materialize the sparse data
data_dense = data_vectorized.todense()
# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
#GridSearch the best LDA model?
# Define Search Param
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
lda = LatentDirichletAllocation()
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)
# Best Model
best_lda_model = model.best_estimator_
print(best_lda_model)
# Model Parameters
print("Best Model's Params: ", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
#Scree graph (elbow method)
# Cluster Number optimization using LDA Coherence Score
dictionary=corpora.Dictionary(text_corpus)
corpus=[dictionary.doc2bow(text) for text in text_corpus]
coherence_values = []
model_list = []
i_model_list = []
i_cv_list = []
limit=21
start=2
step=1

for num_topics in range(start, limit, step):
            print('Number of Topics', num_topics)
            model = gensim.models.LdaModel(corpus,num_topics=num_topics, id2word=dictionary,random_state=100,chunksize=10000,passes=2,alpha='auto')
            
            coherencemodel = CoherenceModel(model=model, texts=text_corpus,corpus=corpus,dictionary=dictionary, coherence='c_v')
            print(' CV ' , coherencemodel.get_coherence())
            coherence_values.append(coherencemodel.get_coherence())
            model_list.append(num_topics)
    
           
get_ipython().run_line_magic('matplotlib', 'inline')
limit=21
start=2
step=1
x=range(start,limit,step)
plt.plot(x,coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence Score")
plt.legend("Coherence_values",loc='best')
plt.grid()
work_dir=os.getcwd()
title_name="TFIDF_20_newsgroup_elbow_plot"
img_name=str(work_dir)+'/'+str(title_name)+str("_")+str(dt_stamp)+ str(".png")
#fig=plt.figure(figsize=(6,4))
plt.savefig(img_name,dpi=300,bbox_inches='tight')
plt.show()
plt.clf()
```
#### Generated TF-IDF by using TfidfVectorizer from (Gensim)
```python
bow_corpus = [dictionary.doc2bow(text) for text in text_corpus]
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
dictionary=corpora.Dictionary(text_corpus)

coherence_values = []
model_list = []
i_model_list = []
i_cv_list = []
limit=21
start=2
step=1
for num_topics in range(start, limit, step):
            print('Number of Topics', num_topics)            
            model = gensim.models.LdaModel(corpus_tfidf,num_topics=num_topics, id2word=dictionary,random_state=100,chunksize=10000,passes=2,alpha='auto', eta='auto')
            
            coherencemodel = CoherenceModel(model=model, texts=text_corpus,corpus=corpus_tfidf,dictionary=dictionary, coherence='c_v')
            print(' CV ' , coherencemodel.get_coherence())
            coherence_values.append(coherencemodel.get_coherence())
            model_list.append(num_topics)
dt_stamp = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H_%M_%S')    
limit=21
start=2
step=1           
get_ipython().run_line_magic('matplotlib', 'inline')
x=range(start,limit,step)
plt.plot(x,coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence Score")
plt.legend("Coherence_values",loc='best')
plt.grid()
work_dir=os.getcwd()
title_name="TFIDF_GENSIM_20_newsgroup_elbow_plot"
img_name=str(work_dir)+'/'+str(title_name)+str("_")+str(dt_stamp)+ str(".png")
#fig=plt.figure(figsize=(6,4))
plt.savefig(img_name,dpi=300,bbox_inches='tight')
plt.show()
plt.clf()
```
