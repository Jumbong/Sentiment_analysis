"""
Étape 5 — Classification et reproduction de la Table 2
Article section 4.3 : "Document Polarity Classification"

Pipeline :
  1. Construire les features de chaque document
       - Bag of Words (bnc) : vecteur tf normalisé cosinus
       - Our model          : R × v  (produit matriciel)
       - Combined           : [R × v || v] concaténation
  2. Entraîner un SVM linéaire sur le train
  3. Évaluer sur le test
  4. Reproduire les chiffres de la Table 2
"""

import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

# ─── Constantes ───────────────────────────────────────────────────────────────

# Pondération 'bnn' — article section 4.3
# b = binary tf (1 si présent, sinon 0) — NON, bnn = tf brut non normalisé
# n = pas de idf
# n = pas de normalisation sur v avant le produit Rv
# La normalisation cosinus est appliquée APRÈS le produit Rv

# ─── Bag of Words ─────────────────────────────────────────────────────────────

def build_bow(text, word2idx):
    """
    Construit le vecteur bag-of-words 'bnc' d'un document.
    b = tf binaire (1 si présent)
    n = pas de idf
    c = normalisation cosinus

    Article section 4.2 : "bnc weighting"
    """
    vec = np.zeros(len(word2idx))
    for word in text.split():
        if word in word2idx:
            vec[word2idx[word]] = 1.0          # b = binaire
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec     # c = cosinus

def build_bnn(text, word2idx):
    """
    Construit le vecteur bag-of-words 'bnn' d'un document.
    b = tf binaire
    n = pas de idf
    n = pas de normalisation

    Article section 4.3 : utilisé pour le produit Rv
    """
    vec = np.zeros(len(word2idx))
    for word in text.split():
        if word in word2idx:
            vec[word2idx[word]] = 1.0
    return vec

# ─── Features document ────────────────────────────────────────────────────────

def doc_features_bow(reviews, word2idx):
    """Bag of Words bnc — baseline de l'article."""
    return np.array([build_bow(r.text, word2idx) for r in reviews])

def doc_features_model(reviews, R, word2idx):
    """
    Features du modèle : R × v avec normalisation cosinus finale.
    Article section 4.3 : "we obtain features using a matrix-vector product Rv"
    """
    features = []
    for r in reviews:
        v      = build_bnn(r.text, word2idx)   # vecteur bnn  (5000,)
        Rv     = R @ v                          # produit      (50,)
        norm   = np.linalg.norm(Rv)
        Rv     = Rv / norm if norm > 0 else Rv  # normalisation cosinus
        features.append(Rv)
    return np.array(features)

def doc_features_combined(reviews, R, word2idx):
    """
    Concaténation [Rv || BoW] — meilleure configuration de l'article.
    Article section 4.3 : "we evaluate performance of the two feature
    representations concatenated"
    """
    bow   = doc_features_bow(reviews, word2idx)    # (N, 5000)
    model = doc_features_model(reviews, R, word2idx)  # (N, 50)
    return np.hstack([model, bow])                 # (N, 5050)

# ─── Labels ───────────────────────────────────────────────────────────────────

def get_labels(reviews):
    """0 = négatif, 1 = positif."""
    return np.array([1 if r.label >= 0.5 else 0 for r in reviews])

# ─── SVM ──────────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, C=1.0):
    """
    SVM linéaire avec LIBLINEAR — exactement comme l'article section 4.3.
    C = paramètre de régularisation (même valeur que Pang & Lee 2004).
    """
    svm = LinearSVC(C=C, max_iter=10000, random_state=42)
    svm.fit(X_train, y_train)
    return svm

def evaluate(svm, X_test, y_test):
    """Accuracy en pourcentage."""
    return round(100 * svm.score(X_test, y_test), 2)

# ─── Reproduction Table 2 ─────────────────────────────────────────────────────

def run_table2(data, word2idx):
    """
    Reproduit les résultats de la Table 2 de l'article.
    Colonne 'Our Dataset' (IMDB 50 000).
    """
    train = data['train_labeled']   # 25 000 critiques
    test  = data['test']            # 25 000 critiques

    y_train = get_labels(train)
    y_test  = get_labels(test)

    # Charger les matrices R de l'étape 3 et 4
    R_sem  = np.load('./R_semantic.npy')
    R_full = np.load('./R_full.npy')

    print("Construction des features...")
    results = {}

    # ── Configuration 1 : Bag of Words seul ──
    print("  [1/4] Bag of Words (bnc)...")
    X_train = doc_features_bow(train, word2idx)
    X_test  = doc_features_bow(test,  word2idx)
    svm     = train_svm(X_train, y_train)
    results['Bag of Words (bnc)'] = evaluate(svm, X_test, y_test)

    # ── Configuration 2 : Semantic Only ──
    print("  [2/4] Our Semantic Only...")
    X_train = doc_features_model(train, R_sem, word2idx)
    X_test  = doc_features_model(test,  R_sem, word2idx)
    svm     = train_svm(X_train, y_train)
    results['Our Semantic Only'] = evaluate(svm, X_test, y_test)

    # ── Configuration 3 : Full model ──
    print("  [3/4] Our Full...")
    X_train = doc_features_model(train, R_full, word2idx)
    X_test  = doc_features_model(test,  R_full, word2idx)
    svm     = train_svm(X_train, y_train)
    results['Our Full'] = evaluate(svm, X_test, y_test)

    # ── Configuration 4 : Full + Bag of Words ──
    print("  [4/4] Our Full + Bag of Words (bnc)...")
    X_train = doc_features_combined(train, R_full, word2idx)
    X_test  = doc_features_combined(test,  R_full, word2idx)
    svm     = train_svm(X_train, y_train)
    results['Our Full + BoW'] = evaluate(svm, X_test, y_test)

    return results

# ─── Affichage ────────────────────────────────────────────────────────────────

def print_table2(results):
    """Affiche les résultats au format Table 2 de l'article."""
    print(f"""
{'='*55}
TABLE 2 — CLASSIFICATION ACCURACY (dataset IMDB)
{'='*55}
{'Méthode':<30} {'Notre résultat':>15} {'Article':>8}
{'─'*55}
{'Bag of Words (bnc)':<30} {results['Bag of Words (bnc)']:>14}%  {'87.80%':>7}
{'Our Semantic Only':<30} {results['Our Semantic Only']:>14}%  {'87.30%':>7}
{'Our Full':<30} {results['Our Full']:>14}%  {'87.44%':>7}
{'Our Full + BoW':<30} {results['Our Full + BoW']:>14}%  {'88.89%':>7}
{'='*55}
""")

# ─── Main ─────────────────────────────────────────────────────────────────────

