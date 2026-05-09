
from sentiment_vectors.data import load_imdb, print_stats


if __name__ == '__main__':
    DATA_DIR = './aclImdb'
    data = load_imdb(DATA_DIR)
    print_stats(data)
    print("[OK] Prêt pour l'étape 2.")
    