import os
import numpy as npp
from sklearn.feature_extraction.text import CountVectorizer
import re
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# save numpy array as csv file
from numpy import savetxt
# define data
# save to csv file
import json
import contractions
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import spacy
from collections import Counter, defaultdict
import seaborn as sns
from wordcloud import WordCloud





def input_docs(folder):
    documents = []
    html_tag_pattern = re.compile(r'<.*?>')  # pattern clair pour retirer toutes les balises HTML
    
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            # Supprimer explicitement toutes les balises HTML
            text = re.sub(html_tag_pattern, ' ', text)
            
            # Nettoyage supplémentaire : retirer caractères non-alphabétiques et espaces superflus
            words = [word.strip() for word in re.split(r"[^a-z]+", text) if word.strip()]
            cleaned_text = ' '.join(words)
            
            documents.append(cleaned_text)
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
# je veux un motif qui recherche "n" suivi d'un espace suivi d'un "t" suivi d'un espace.

def process_doc(doc):
    """
    Traite un document (chaîne de caractères) en :
      - Vérifiant si le document contient une contraction mal formée (ex. "don t").
      - Si oui, recolle les tokens ("don" + "t" → "don't") et étend la contraction en "do not".
      - Sinon, retourne le texte tokenisé normalement.
    """
    # Vérifier la présence du motif : mot se terminant par "n", espace, "t"
    if re.sub(r"\b(\w+n)\st\s", r"\1't ", doc, flags=re.IGNORECASE):
        # Recolle les tokens avec une apostrophe : "don t" devient "don't"
        text_fixed = re.sub(r"\b(\w+n)\st\s", r"\1't ", doc, flags=re.IGNORECASE)
        # Appliquer contractions.fix() pour transformer "don't" en "do not"
        text_expanded = contractions.fix(text_fixed)
        return text_expanded.split()
    else:
        # Si le motif n'est pas trouvé, on ne fait rien et on tokenize normalement
        return doc.split()



def word_vectors(neg,pos):
    vectors=[None]*len(neg)+[None]*len(pos)
    for i,doc in enumerate(neg):
        
        words= process_doc(doc)
        vector=[None]*len(words)
        for j,word in enumerate(words):
            vector[j]=word
        vectors[i]=vector
    
    for i,doc in enumerate(pos):
        words= process_doc(doc)
        vector=[None]*len(words)
        for j,word in enumerate(words):
            vector[j]=word
        vectors[i+len(neg)]=vector
    return vectors
            
def pos_tag_docs(docs, batch_size=50, n_process=1):
    """
    docs : liste de documents, 
           chaque document étant une liste de tokens (mots).
    Retourne :
      - pos_counts : un Counter donnant la fréquence de chaque tag (ADJ, NOUN, etc.)
      - tokens_by_pos : un dict de Counters 
                        { 'ADJ': {'great': 10, 'bad': 5, ...}, 'NOUN': {...}, ... }
    """
    pos_counts = Counter()
    tokens_by_pos = defaultdict(Counter)
    
    # Reconstituer tous les textes à partir de la liste de tokens
    texts = [" ".join(token_list) for token_list in docs]
    
    # Utiliser nlp.pipe pour traiter tous les documents en batch
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        for token in doc:
            pos_tag = token.pos_       # ex. "NOUN", "VERB", "ADJ", etc.
            surface_form = token.text  # ou token.lemma_ pour la forme lemmatisée
            pos_counts[pos_tag] += 1
            tokens_by_pos[pos_tag][surface_form.lower()] += 1
            
    return pos_counts, tokens_by_pos

def plot_top_tokens_for_tag(tokens_by_pos, tag, top_n=10):
    counter = tokens_by_pos.get(tag, None)
    if not counter:
        print(f"Aucun token trouvé pour le tag {tag}")
        return
    
    most_common = counter.most_common(top_n)
    tokens, freqs = zip(*most_common)
    
    plt.figure(figsize=(8,4))
    plt.barh(tokens, freqs, color='green')
    plt.gca().invert_yaxis()
    plt.xlabel("Fréquence")
    plt.ylabel("Tokens")
    plt.title(f"Top {top_n} tokens pour le tag {tag}")
    plt.show()


class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, batch_size=50, n_process=1, excluded_tags=None):
        self.batch_size = batch_size
        self.n_process = n_process
        # Définir les tags à exclure (par défaut, aucun n'est exclu)
        if excluded_tags is None:
            self.excluded_tags = set()
        else:
            self.excluded_tags = set(excluded_tags)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X est une liste de documents, chaque document étant une liste de tokens
        docs_text = [" ".join(doc) for doc in X]
        lemmatized_docs = [None] * len(docs_text)
        for i, doc in enumerate(nlp.pipe(docs_text, batch_size=self.batch_size, n_process=self.n_process)):
            # On filtre les tokens dont le pos tag est dans excluded_tags
            lemmatized = " ".join([token.lemma_ for token in doc 
                                   if token.pos_ not in self.excluded_tags])
            lemmatized_docs[i] = lemmatized
        return lemmatized_docs
    
          
def lemmatizProcess(docs, excluded_tags=None):
    
    excluded_set = set(excluded_tags) if excluded_tags else set()
    
    # Préallouer le résultat avec la bonne taille
    lemmatized_docs = [None] * len(docs)
    
    # Si les documents sont déjà tokenisés mais nécessitent un traitement spaCy
    docs_text = [" ".join(doc) for doc in docs]
    
    # Traiter les documents avec plus de processus si possible
    for i, doc in enumerate(nlp.pipe(docs_text, batch_size=50, n_process=-1)):
        # Filtrer les tokens dont le pos tag est dans excluded_set
        lemmatized = [token.lemma_ for token in doc if token.pos_ not in excluded_set]
        lemmatized_docs[i] = lemmatized
    
    return lemmatized_docs
    
   
def document_vector(doc_tokens, model):
    # On récupère les vecteurs pour les tokens présents dans le vocabulaire du modèle
    vectors = [model.wv[token] for token in doc_tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Retourne un vecteur zéro si aucun mot n'est reconnu
        return np.zeros(model.vector_size)
    





def compute_roc(pipelines, X_test, y_test):
    roc_results = {}

    for name, pipeline in pipelines.items():
        if name == 'SVM':
            proba = pipeline.decision_function(X_test)
        else:
            proba = pipeline.predict_proba(X_test)[:, 1]  # probabilité classe positive
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

    return roc_results

def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))