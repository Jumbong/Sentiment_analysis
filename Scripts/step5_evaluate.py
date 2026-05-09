
from sentiment_vectors.data import load_imdb
from sentiment_vectors.vocabulary import build_vocab
from sentiment_vectors.evaluation import run_table2, print_table2

if __name__ == '__main__':


    data             = load_imdb('./aclImdb')
    word2idx, idx2word = build_vocab(data)

    print("Reproduction de la Table 2...\n")
    results = run_table2(data, word2idx)
    print_table2(results)