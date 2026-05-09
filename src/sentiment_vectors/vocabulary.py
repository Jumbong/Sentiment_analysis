"""
Étape 2 — Construction du vocabulaire
Article section 4.1 :
  - Vocabulaire construit sur les 75 000 critiques d'entraînement
  - On ignore les 50 mots les plus fréquents (trop grammaticaux)
  - On garde les 5 000 suivants
  - Chaque mot reçoit un index entier (le modèle travaille avec des indices)
"""

from collections import Counter

# ─── Constantes (hyperparamètres de l'article) ────────────────────────────────

N_SKIP  = 50    # mots les plus fréquents à ignorer
N_VOCAB = 5000  # taille du vocabulaire final

# ─── Construction ─────────────────────────────────────────────────────────────

def build_vocab(data):
    """
    Construit le vocabulaire à partir des 75 000 critiques d'entraînement.

    Retourne deux dicts :
      word2idx : { mot   → index }  ex: {'film': 0, 'great': 1, ...}
      idx2word : { index → mot   }  ex: {0: 'film', 1: 'great', ...}
    """
    # Les 75 000 critiques d'entraînement = labellisées + non labellisées
    train_reviews = data['train_labeled'] + data['train_unlabeled']

    # Compter tous les mots sur l'ensemble du train
    # Counter({'the': 2000000, 'film': 350000, ...})
    counts = Counter(
        word
        for review in train_reviews
        for word   in review.text.split()
    )

    # most_common() retourne [(mot, count), ...] trié par fréquence
    # [N_SKIP : N_SKIP + N_VOCAB] = on saute les 50 premiers, on prend les 5000 suivants
    vocab = [word for word, _ in counts.most_common()[N_SKIP : N_SKIP + N_VOCAB]]

    # Index entier pour chaque mot — le modèle n'utilise que des entiers
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def print_vocab_stats(counts, word2idx):
    """Affiche les statistiques du vocabulaire construit."""
    all_words   = counts.most_common()
    top_50      = [w for w, _ in all_words[:N_SKIP]]
    in_vocab    = [w for w, _ in all_words[N_SKIP : N_SKIP + N_VOCAB]]
    too_rare    = [w for w, _ in all_words[N_SKIP + N_VOCAB:]]

    print(f"""
{'='*45}
STATISTIQUES DU VOCABULAIRE
{'='*45}
Mots uniques total   : {len(all_words):>8}
Mots ignorés (top50) : {len(top_50):>8}
Vocabulaire retenu   : {len(word2idx):>8}
Mots trop rares      : {len(too_rare):>8}

Top 5 mots ignorés (trop fréquents) :
  {top_50[:5]}

Top 5 mots du vocabulaire :
  {in_vocab[:5]}

Exemples de mots de sentiment dans le vocab :
  { {w: word2idx[w] for w in ['wonderful','terrible','awful','amazing','boring'] if w in word2idx} }
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

