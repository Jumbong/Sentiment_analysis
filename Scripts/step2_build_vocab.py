from sentiment_vectors.data import load_imdb
from sentiment_vectors.vocabulary import build_vocab, print_vocab_stats
from collections import Counter

if __name__ == '__main__':

    data             = load_imdb('./aclImdb')
    print(data['train_labeled'][0])
    word2idx, idx2word = build_vocab(data)

    # Reconstruire counts pour les stats
    train_reviews = data['train_labeled'] + data['train_unlabeled']
    counts        = Counter(w for r in train_reviews for w in r.text.split())

    print_vocab_stats(counts, word2idx)
    print(f"[OK] Vocabulaire construit : {len(word2idx)} mots.")