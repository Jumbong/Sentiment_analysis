"""
Étape 1 — Chargement des données IMDB
Article section 4.1 :
  - 25 000 critiques labellisées  (train/pos + train/neg)
  - 50 000 critiques non labellisées (train/unsup)
  - 25 000 critiques de test       (test/pos  + test/neg)

Nettoyage minimal fidèle à l'article :
  ✓ minuscules + suppression balises HTML
  ✗ pas de stemming, pas de stopwords, "!" et ":-)" conservés
"""

import os
import re
from collections import namedtuple

# ─── Types ────────────────────────────────────────────────────────────────────

# Un document = une critique IMDB
# label : float [0,1] si labellisé, None si unsup
# bucket : 'train_labeled' | 'train_unlabeled' | 'test'
Review = namedtuple('Review', ['text', 'stars', 'label', 'bucket'])

# ─── Constantes ───────────────────────────────────────────────────────────────

HTML     = re.compile(r'<[^>]+>')
STARS_RE = re.compile(r'\d+_(\d+)\.txt')
SPLITS   = [('train', 'pos'), ('train', 'neg'), ('train', 'unsup'),
            ('test',  'pos'), ('test',  'neg')]

# ─── Fonctions pures ──────────────────────────────────────────────────────────

# def clean(text):
#     """Minuscules + suppression HTML. Conserve '!' et ':-)'."""
#     return re.sub(r'\s+', ' ', re.sub(HTML, ' ', text.lower())).strip()


def clean(text):
    text = re.sub(HTML, ' ', text.lower())
    text = re.sub(r'(\w)[.,;:!?]+(\s|$)', r'\1 ', text)  # "awful," → "awful"
    return re.sub(r'\s+', ' ', text).strip()

def to_label(stars):
    """Étoiles (1-10) → probabilité [0,1]. Article section 4.1."""
    return (stars - 1) / 9.0

def extract_stars(fname):
    """'1234_7.txt' → 7. Retourne None si format invalide."""
    m = STARS_RE.match(fname)
    return int(m.group(1)) if m else None

def to_bucket(split, category):
    """Détermine dans quel bucket ranger la critique."""
    if category == 'unsup': return 'train_unlabeled'
    if split    == 'train': return 'train_labeled'
    return 'test'

def read_folder(data_dir, split, category):
    """Lit tous les .txt d'un dossier → liste de Review."""
    folder = os.path.join(data_dir, split, category)
    fnames = [f for f in os.listdir(folder) if f.endswith('.txt')]
    print(f"  {split}/{category:5s} : {len(fnames):>6} fichiers")
    return [
        Review(
            text   = clean(open(os.path.join(folder, f), encoding='utf-8').read()),
            stars  = extract_stars(f),
            label  = to_label(extract_stars(f)) if extract_stars(f) else None,
            bucket = to_bucket(split, category),
        )
        for f in fnames
    ]

# ─── Chargement principal ─────────────────────────────────────────────────────

def load_imdb(data_dir):
    """
    Charge toutes les critiques IMDB.
    Retourne un dict : { bucket → list[Review] }
    """
    print("Chargement IMDB...")
    all_reviews = [r for split, cat in SPLITS for r in read_folder(data_dir, split, cat)]
    buckets     = ['train_labeled', 'train_unlabeled', 'test']
    return {b: [r for r in all_reviews if r.bucket == b] for b in buckets}

# ─── Stats ────────────────────────────────────────────────────────────────────

def print_stats(data):
    labeled, unlabeled, test = (
        data['train_labeled'],
        data['train_unlabeled'],
        data['test'],
    )
    labels = [r.label for r in labeled]
    print(f"""
{'='*45}
STATISTIQUES DU DATASET
{'='*45}
Train labellisé     : {len(labeled):>6}
  dont pos (≥7★)   : {sum(1 for r in labeled if r.stars >= 7):>6}
  dont neg (≤4★)   : {sum(1 for r in labeled if r.stars <= 4):>6}
Train non labellisé : {len(unlabeled):>6}
Test                : {len(test):>6}

Labels — min:{min(labels):.3f}  max:{max(labels):.3f}  moy:{sum(labels)/len(labels):.3f}
{'─'*45}
Exemple : {next(r for r in labeled if r.stars >= 7).text[:120]}...
""")

# ─── Main ─────────────────────────────────────────────────────────────────────

