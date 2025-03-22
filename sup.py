import os
import numpy as np
from scipy.special import logsumexp
import re
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import random
# save numpy array as csv file
from numpy import savetxt
# define data
# save to csv file
import json

def input_docs(folder):
    documents=[]
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            string=''
            for line in file:
                line=line.strip() # strip() removes all trailing and starting whitespaces as "   word  " becomes "word"                    
                for word in re.split(r"[^A-Za-z]+", line):          #split on any of the character except A-Z,a-z
                    word = word.lower().strip()
                    if len(word) == 0:
                        continue
                    string=string+' '+word
 
            documents.append(string)
    return documents

def freq(docs):
    strings={}
    i=1
    for doc in docs:
        words=doc.split()
        for word in words:
            if word not in strings:
                strings[word]=[i]
            else:
                if strings[word][-1]!=i:
                    strings[word].append(i)
        i+=1
    words={}
    for key in strings.keys():
        j=len(strings[key])
        if j < 2100 and j >20:
            words[key]=j
    with open('dict.json', 'w') as f:
        f.write(json.dumps(words))
    return words

def word_vectors(neg,pos):
    vectors=[]
    for doc in neg:
        words=doc.split()
        vector=[]
        for word in words:
            vector.append(word)
        vectors.append(vector)
    
    for doc in pos:
        words=doc.split()
        vector=[]
        for word in words:
            vector.append(word)
        vectors.append(vector)
    
    return vectors


folder_path = 'MoveReview\\aclImdb\\train\\neg'
documents = input_docs(folder_path)
neg=freq(documents)
docs_pos=input_docs('pos')
pos=freq(docs_pos)
docs=word_vectors(documents,docs_pos)
model = Word2Vec(docs, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
print('done')
word_vectors = []
sentiment_labels=[]

for word in neg:
    if word not in model.wv:
        continue
    word_vector = model.wv[word]
    word_vectors.append(word_vector)
    sentiment_labels.append('neg')

for word in pos:
    if word not in model.wv:
        continue
    word_vector = model.wv[word]
    word_vectors.append(word_vector)
    sentiment_labels.append('pos')

print(len(sentiment_labels))
while True:
   
    # for doc in docs_pos:
    #    words=doc.split()
    #    for word in words:
    #        if word not in model.wv:
    #            continue
    #        word_vector = model.wv[word]
    #        word_vectors.append(word_vector)
    #        sentiment_labels.append('pos')
    # for doc in documents:
    #    words=doc.split()
    #    for word in words:
    #        if word not in model.wv:
    #            continue
    #        word_vector = model.wv[word]
    #        word_vectors.append(word_vector)
    #        sentiment_labels.append('neg')
    # for i in range(1,22000):
    #     l=random.randint(0,14367)
    #     m=random.randint(0,14367)
    #     word_vectors[l],word_vectors[m]=word_vectors[m],word_vectors[l]
    #     sentiment_labels[l],sentiment_labels[m]=sentiment_labels[m],sentiment_labels[l]

    logreg = LogisticRegression(max_iter=2000)
    word_vectors=np.array(word_vectors)
    sentiment_labels=np.array(sentiment_labels)
    logreg.fit(word_vectors, sentiment_labels)
    # word_vector_new = model.wv['bad']# New word vector for prediction

    # predicted_sentiment = logreg.predict([word_vector_new])
    # print(predicted_sentiment)
    # predicted_probabilities = logreg.predict_proba([word_vector_new])
    # probability_positive_sentiment = predicted_probabilities[0, 1]
    # print(probability_positive_sentiment)
    # x=probability_positive_sentiment
    # word_vector_new = model.wv['good']# New word vector for prediction
    # predicted_sentiment = logreg.predict([word_vector_new])
    # print(predicted_sentiment)
    # predicted_probabilities = logreg.predict_proba([word_vector_new])
    # probability_positive_sentiment = predicted_probabilities[0, 1]
    # print(probability_positive_sentiment)
    # if probability_positive_sentiment > 0.55 and x<0.35 and x>0.34 and probability_positive_sentiment<0.56 or True:
        # print(word_vectors)
        # print(sentiment_labels)
    np.save('vectors1.txt', word_vectors,) # save
    np.save('word1.txt', sentiment_labels) # save

    break
    
    
