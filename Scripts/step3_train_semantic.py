from sentiment_vectors.data import load_imdb
from sentiment_vectors.vocabulary import build_vocab
from sentiment_vectors.semantic import train_semantic, most_similar
import numpy as np


if __name__ == '__main__':
    
    BETA = 50  # Dimensionalité des vecteurs sémantiques

    data             = load_imdb('./aclImdb')
    word2idx, idx2word = build_vocab(data)

    print(f"Vocabulaire : {len(word2idx)} mots | Dimensions : {BETA}")
    print("Début de l'entraînement sémantique...\n")

    R, b = train_semantic(data, word2idx)

    # Évaluation qualitative — reproduit Table 1 de l'article
    print("\n" + "="*45)
    print("MOTS LES PLUS SIMILAIRES (Table 1)")
    print("="*45)
    for word in ['wonderful', 'terrible', 'boring', 'romantic']:
        similars = most_similar(word, R, word2idx, idx2word)
        print(f"\n{word:>12} → {similars}")

    # Sauvegarder R et b pour les étapes suivantes
    np.save('./R_semantic.npy', R)
    np.save('./b_semantic.npy', b)
    print("\n[OK] R et b sauvegardés.")