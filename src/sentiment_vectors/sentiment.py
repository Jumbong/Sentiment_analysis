"""
Étape 4 — Modèle complet : sémantique + sentiment
Article section 3.2 : "Capturing Word Sentiment"

Nouveaux objets par rapport à l'étape 3 :
  ψ  (psi) : vecteur (50,)  — poids de la régression logistique
  bc       : scalaire       — biais du classifieur sentiment

Objectif complet (eq. 11 de l'article) :
  ν||R||²_F
  + Σ_k λ||θ̂_k||²
  + Σ_k Σ_i log p(w_i|θ̂_k; R, b)
  + Σ_k (1/|S_k|) Σ_i log p(s_k|w_i; R, ψ, bc)

  |S_k| = nombre de documents de la même classe que k
        = 12 500 pour les positifs (s_k >= 0.5)
        = 12 500 pour les négatifs (s_k  < 0.5)

L'entraînement est identique à l'étape 3 (alternating maximization)
avec une mise à jour supplémentaire de ψ et bc à chaque document.
"""

import numpy as np

# ─── Hyperparamètres ──────────────────────────────────────────────────────────

BETA         = 50
LAMBDA       = 0.1
NU           = 0.1
LR_THETA     = 0.01
LR_R         = 0.00001   # plus petit pour préserver la sémantique
#LR_PSI       = 0.001    # plus grand pour apprendre le sentiment
N_ITER_THETA = 20
N_EPOCHS     = 15
LR_PSI = 10.0 

# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def sigmoid(x):
    """σ(x) = 1 / (1 + exp(-x)). Clippé pour stabilité numérique."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def log_softmax(theta, R, b):
    """log p(w|θ) pour tous les mots du vocabulaire. Identique étape 3."""
    logits  = theta @ R + b
    logits -= logits.max()
    return logits - np.log(np.sum(np.exp(logits)))

def to_word_ids(text, word2idx):
    """Texte → liste d'indices vocabulaire. Mots hors vocab ignorés."""
    return [word2idx[w] for w in text.split() if w in word2idx]

# ─── Initialisation ───────────────────────────────────────────────────────────

def init_model(vocab_size, beta=BETA):
    """
    Charge R et b de l'étape 3 si disponibles.
    Initialise ψ et bc à zéro (pas encore de signal sentiment).
    """
    try:
        R  = np.load('./R_semantic.npy')
        b  = np.load('./b_semantic.npy')
        print("[OK] R et b chargés depuis l'étape 3.")
    except FileNotFoundError:
        R  = np.random.randn(beta, vocab_size) * 0.01
        b  = np.zeros(vocab_size)
        print("[WARN] R et b initialisés aléatoirement.")

    # ψ et bc initialisés à zéro — l'article section 3.2
    psi = np.zeros(beta)
    bc  = 0.0
    return R, b, psi, bc

# ─── Étape A : estimation MAP de θ_k (inchangée) ─────────────────────────────

def estimate_theta(word_ids, R, b):
    """Identique à l'étape 3 — θ_k ne dépend pas du sentiment."""
    theta = np.zeros(BETA)
    for _ in range(N_ITER_THETA):
        log_probs = log_softmax(theta, R, b)
        probs     = np.exp(log_probs)
        observed  = R[:, word_ids].sum(axis=1)
        expected  = R @ probs
        N         = len(word_ids)
        grad      = observed - N * expected - 2 * LAMBDA * theta
        grad      = np.clip(grad, -1.0, 1.0)
        theta    += LR_THETA * grad
    return theta

# ─── Étape B : mise à jour de R, b, ψ, bc ────────────────────────────────────

def update_semantic(theta, word_ids, R, b):
    """Mise à jour sémantique — identique à l'étape 3."""
    log_probs = log_softmax(theta, R, b)
    probs     = np.exp(log_probs)
    N         = len(word_ids)
    counts    = np.zeros(len(b))
    for w in word_ids:
        counts[w] += 1
    residual  = np.clip(counts - N * probs, -1.0, 1.0)
    R        += LR_R * (np.outer(theta, residual) - 2 * NU * R)
    b        += LR_R * residual
    return R, b

def update_sentiment(word_ids, label, S_k, R, psi, bc):
    """
    Mise à jour sentiment — équation 11 de l'article.

    Pour chaque mot w du document de label s_k :
      p(s=1|w) = σ(ψᵀ φ_w + bc)
      erreur   = (1/|S_k|) * (s_k - p(s=1|w))

    |S_k| = nombre de documents de la même classe dans le dataset
          → 12 500 si positif (s_k >= 0.5)
          → 12 500 si négatif (s_k  < 0.5)

    Ce terme 1/|S_k| est crucial — il empêche la classe majoritaire
    de dominer l'apprentissage du sentiment (eq. 11, section 3.3).

    Gradients :
      ∂L/∂ψ    = erreur * φ_w
      ∂L/∂bc   = erreur
      ∂L/∂φ_w  = erreur * ψ
    """
    weight = 1.0 / S_k                            # ← 1/|S_k| de l'article
    for w in word_ids:
        phi_w   = R[:, w]                          # vecteur du mot w  (50,)
        p_pos   = sigmoid(psi @ phi_w + bc)        # p(s=1|w)
        error   = weight * (label - p_pos)         # pondéré par 1/|S_k|

        # Mise à jour de ψ et bc
        psi    += LR_PSI * error * phi_w
        bc     += LR_PSI * error

        # Mise à jour de φ_w via le sentiment
        R[:, w] += LR_PSI * error * psi

    return R, psi, bc

# ─── Entraînement complet ─────────────────────────────────────────────────────

def train_full(data, word2idx, n_epochs=N_EPOCHS):
    """
    Entraîne le modèle complet (sémantique + sentiment).

    Seules les critiques LABELLISÉES contribuent à la mise à jour sentiment.
    Toutes les critiques (75 000) contribuent à la mise à jour sémantique.
    """
    vocab_size    = len(word2idx)
    R, b, psi, bc = init_model(vocab_size)

    labeled   = data['train_labeled']    # 25 000 — sémantique + sentiment
    unlabeled = data['train_unlabeled']  # 50 000 — sémantique uniquement
    all_train = labeled + unlabeled
    n_docs    = len(all_train)

    # |S_k| — nombre de documents par classe (eq. 11, section 3.3)
    S_pos = sum(1 for r in labeled if r.label >= 0.5)  # critiques positives
    S_neg = sum(1 for r in labeled if r.label  < 0.5)  # critiques négatives
    print(f"  |S_pos| = {S_pos} | |S_neg| = {S_neg}")

    for epoch in range(n_epochs):
        total_ll = 0.0
        np.random.shuffle(all_train)

        for i, review in enumerate(all_train):
            word_ids = to_word_ids(review.text, word2idx)
            if not word_ids:
                continue

            # Étape A — estimer θ_k (identique étape 3)
            theta = estimate_theta(word_ids, R, b)

            # Étape B.1 — mise à jour sémantique (identique étape 3)
            R, b  = update_semantic(theta, word_ids, R, b)

            # Étape B.2 — mise à jour sentiment avec 1/|S_k| (eq. 11)
            if review.label is not None:
                S_k        = S_pos if review.label >= 0.5 else S_neg
                R, psi, bc = update_sentiment(word_ids, review.label, S_k, R, psi, bc)

            # Log-vraisemblance sémantique pour suivi
            log_probs = log_softmax(theta, R, b)
            ll        = sum(log_probs[w] for w in word_ids)
            if np.isfinite(ll):
                total_ll += ll

            if (i + 1) % 5000 == 0:
                print(f"  Epoch {epoch+1} | doc {i+1}/{n_docs} "
                      f"| log-lik moy : {total_ll/(i+1):.2f}")

        print(f"Epoch {epoch+1}/{n_epochs} terminée "
              f"| log-lik moy : {total_ll/n_docs:.2f}")

    return R, b, psi, bc

# ─── Évaluation qualitative ───────────────────────────────────────────────────

def most_similar(word, R, word2idx, idx2word, top_n=5):
    """Mots les plus similaires par cosinus — Table 1 colonne gauche."""
    if word not in word2idx:
        return f"'{word}' hors vocabulaire"
    idx    = word2idx[word]
    phi_w  = R[:, idx]
    norms  = np.linalg.norm(R, axis=0)
    sims   = (R.T @ phi_w) / (norms * np.linalg.norm(phi_w) + 1e-8)
    top_ids = [i for i in np.argsort(sims)[::-1] if i != idx][:top_n]
    return [(idx2word[i], round(float(sims[i]), 3)) for i in top_ids]

def sentiment_score(word, R, word2idx, psi, bc):
    """
    Score de sentiment d'un mot = σ(ψᵀ φ_w + bc).
    > 0.5 → positif, < 0.5 → négatif.
    """
    if word not in word2idx:
        return None
    phi_w = R[:, word2idx[word]]
    return round(float(sigmoid(psi @ phi_w + bc)), 3)

# ─── Main ─────────────────────────────────────────────────────────────────────

