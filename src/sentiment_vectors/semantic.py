"""
Étape 3 — Composante sémantique (non supervisée)
Article section 3.1 : "Capturing Semantic Similarities"

Objets appris :
  R   : matrice (50 x 5000) — vecteurs de mots  φ_w = R[:, w]
  b   : vecteur (5000,)     — biais par mot
  θ_k : vecteur (50,)       — thème du document k (calculé puis oublié)

Algorithme : alternating maximization
  Étape A : fixer R, b → estimer θ_k pour chaque document (MAP)
  Étape B : fixer tous les θ_k → mettre à jour R, b (gradient)
"""

import numpy as np
from collections import Counter

# ─── Hyperparamètres (article section 3.1) ────────────────────────────────────

BETA       = 50      # dimensions des vecteurs de mots
LAMBDA     = 0.1     # régularisation sur θ  (||θ||²)
NU         = 0.1     # régularisation sur R  (||R||²_F)
#LR_THETA   = 0.01    # learning rate pour θ
LR_R       = 0.001   # learning rate pour R et b
#N_ITER_THETA = 10    # itérations MAP par document
#N_EPOCHS     = 5     # passages sur le dataset complet

# Après
LR_THETA     = 0.05
N_ITER_THETA = 20
N_EPOCHS     = 10

# ─── Initialisation ───────────────────────────────────────────────────────────

def init_model(vocab_size, beta=BETA):
    """
    Initialise R et b aléatoirement.
    R ~ N(0, 0.01) : petites valeurs pour éviter la saturation du softmax
    b = 0          : pas de biais initial
    """
    R = np.random.randn(beta, vocab_size) * 0.01   # (50 x 5000)
    b = np.zeros(vocab_size)                        # (5000,)
    return R, b

# ─── Softmax et log-vraisemblance ─────────────────────────────────────────────

def log_softmax(theta, R, b):
    """
    Calcule log p(w|θ) pour tous les mots du vocabulaire.

    log p(w|θ) = θᵀφ_w + b_w - log Σ_w' exp(θᵀφ_w' + b_w')
                 ───────────────   ──────────────────────────
                 énergie du mot w       constante de normalisation

    Retourne un vecteur (vocab_size,) de log-probabilités.
    """
    # Énergie de chaque mot : θᵀR + b → vecteur (5000,)
    logits = theta @ R + b

    # Soustraction du max pour stabilité numérique (évite exp(grands nombres))
    logits -= logits.max()

    # Log-softmax
    return logits - np.log(np.sum(np.exp(logits)))

def doc_log_likelihood(theta, word_ids, R, b):
    """
    Log-vraisemblance d'un document = somme des log p(w|θ) sur ses mots.
    + terme de régularisation sur θ : -λ||θ||²
    """
    log_probs = log_softmax(theta, R, b)                    # (vocab_size,)
    return sum(log_probs[w] for w in word_ids) \
           - LAMBDA * np.dot(theta, theta)

# ─── Étape A : estimation MAP de θ_k ─────────────────────────────────────────

def estimate_theta(word_ids, R, b):
    """
    Trouve θ_k qui maximise la log-vraisemblance du document k.
    Descente de gradient sur θ (R et b sont fixés).

    Gradient de la log-vraisemblance par rapport à θ :
      ∂L/∂θ = Σ_w∈doc φ_w  -  N * E_p[φ_w]  -  2λθ
               ──────────     ────────────────   ────
               mots observés  mots attendus      régul.
    """
    theta = np.zeros(BETA)

    for _ in range(N_ITER_THETA):
        log_probs = log_softmax(theta, R, b)          # (vocab_size,)
        probs     = np.exp(log_probs)                 # (vocab_size,)

        # Gradient : mots observés - mots attendus - régularisation
        # Mots observés : somme des φ_w pour w dans le document
        observed  = R[:, word_ids].sum(axis=1)        # (50,)

        # Mots attendus : espérance sous p(w|θ) = Σ_w p(w|θ) φ_w
        expected  = R @ probs                         # (50,)

        N         = len(word_ids)
        grad      = observed - N * expected - 2 * LAMBDA * theta

        theta    += LR_THETA * grad

    return theta

# ─── Étape B : mise à jour de R et b ─────────────────────────────────────────

def update_R_b(theta, word_ids, R, b):
    """
    Met à jour R et b pour un document, θ étant fixé.

    Gradient par rapport à φ_w (colonne w de R) :
      Si w ∈ document : ∂L/∂φ_w = θ * (count(w) - N*p(w|θ)) - 2ν*φ_w
      Si w ∉ document : ∂L/∂φ_w = θ * (        - N*p(w|θ)) - 2ν*φ_w

    Gradient par rapport à b_w :
      ∂L/∂b_w = count(w) - N * p(w|θ)
    """
    log_probs = log_softmax(theta, R, b)
    probs     = np.exp(log_probs)                     # (vocab_size,)
    N         = len(word_ids)

    # Compter les occurrences de chaque mot dans le document
    counts    = np.zeros(len(b))
    for w in word_ids:
        counts[w] += 1

    # Résidu : différence entre ce qu'on observe et ce qu'on attend
    residual  = counts - N * probs                    # (vocab_size,)

    # Mise à jour de R : gradient = θ ⊗ residual - 2ν*R
    R        += LR_R * (np.outer(theta, residual) - 2 * NU * R)

    # Mise à jour de b : gradient = residual (b non régularisé, article section 3.1)
    b        += LR_R * residual

    return R, b

# ─── Conversion texte → indices ───────────────────────────────────────────────

def to_word_ids(text, word2idx):
    """
    Convertit un texte en liste d'indices vocabulaire.
    Les mots hors vocabulaire sont ignorés.
    """
    return [word2idx[w] for w in text.split() if w in word2idx]

# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_semantic(data, word2idx, n_epochs=N_EPOCHS):
    """
    Entraîne la composante sémantique sur les 75 000 critiques d'entraînement.
    Utilise l'alternating maximization : pour chaque document,
      1. Estimer θ_k  (Étape A)
      2. Mettre à jour R, b avec ce θ_k  (Étape B)
    """
    vocab_size   = len(word2idx)
    R, b         = init_model(vocab_size)

    # 75 000 critiques = labellisées + non labellisées
    all_train    = data['train_labeled'] + data['train_unlabeled']
    n_docs       = len(all_train)

    for epoch in range(n_epochs):
        total_ll = 0.0
        np.random.shuffle(all_train)

        for i, review in enumerate(all_train):
            word_ids = to_word_ids(review.text, word2idx)
            if not word_ids:
                continue

            # Étape A : estimer θ_k pour ce document
            theta = estimate_theta(word_ids, R, b)

            # Étape B : mettre à jour R et b
            R, b  = update_R_b(theta, word_ids, R, b)

            # Log-vraisemblance pour suivi
            total_ll += doc_log_likelihood(theta, word_ids, R, b)

            if (i + 1) % 5000 == 0:
                print(f"  Epoch {epoch+1} | doc {i+1}/{n_docs} "
                      f"| log-lik moy : {total_ll/(i+1):.2f}")

        print(f"Epoch {epoch+1}/{n_epochs} terminée | "
              f"log-lik moy : {total_ll/n_docs:.2f}")

    return R, b

# ─── Évaluation qualitative (Table 1 de l'article) ───────────────────────────

def most_similar(word, R, word2idx, idx2word, top_n=5):
    """
    Retourne les top_n mots les plus similaires à word
    par similarité cosinus — reproduit la Table 1 de l'article.
    """
    if word not in word2idx:
        return f"'{word}' hors vocabulaire"

    idx    = word2idx[word]
    phi_w  = R[:, idx]                              # vecteur du mot cible

    # Similarité cosinus avec tous les mots du vocabulaire
    norms  = np.linalg.norm(R, axis=0)              # (vocab_size,)
    sims   = (R.T @ phi_w) / (norms * np.linalg.norm(phi_w) + 1e-8)

    # Top n (en excluant le mot lui-même)
    top_ids = np.argsort(sims)[::-1]
    top_ids = [i for i in top_ids if i != idx][:top_n]

    return [(idx2word[i], round(float(sims[i]), 3)) for i in top_ids]

# ─── Main ─────────────────────────────────────────────────────────────────────

