from sentiment_vectors.data import load_imdb
from sentiment_vectors.vocabulary import build_vocab
from sentiment_vectors.sentiment import train_full, train_full,BETA ,most_similar, sentiment_score
import numpy as np      

if __name__ == '__main__':
  

    data             = load_imdb('./aclImdb')
    word2idx, idx2word = build_vocab(data)

    print(f"Vocabulaire : {len(word2idx)} mots | Dimensions : {BETA}\n")

    R, b, psi, bc = train_full(data, word2idx)

    # Évaluation qualitative — Table 1 colonne gauche de l'article
    print("\n" + "="*45)
    print("MOTS LES PLUS SIMILAIRES — modèle complet (Table 1)")
    print("="*45)
    for word in ['wonderful', 'terrible', 'boring', 'romantic']:
        print(f"\n{word:>12} → {most_similar(word, R, word2idx, idx2word)}")

    # Score de sentiment — nouveau par rapport à l'étape 3
    print("\n" + "="*45)
    print("SCORES DE SENTIMENT (0=négatif, 1=positif)")
    print("="*45)
    for word in ['wonderful', 'amazing', 'terrible', 'awful', 'boring', 'great']:
        score = sentiment_score(word, R, word2idx, psi, bc)
        barre = '█' * int(score * 20) if score else ''
        print(f"  {word:>12} : {score}  {barre}")

    # Sauvegarder pour l'étape 5
    np.save('./R_full.npy',   R)
    np.save('./b_full.npy',   b)
    np.save('./psi_full.npy', psi)
    np.save('./bc_full.npy',  np.array([bc]))
    print("\n[OK] R, b, ψ, bc sauvegardés.")