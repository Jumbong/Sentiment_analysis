# Sentiment Vectors

A Python reproduction of Learning Word Vectors for Sentiment Analysis by Maas et al. (2011).

This project learns word representations from IMDb movie reviews. The objective is to build word vectors that capture both:

semantic similarity: words used in similar contexts should be close;
sentiment orientation: words carrying similar polarity should also be close.

The implementation follows the main ideas of the paper and evaluates the learned representations using a linear SVM classifier.

## 📁 Project Structure

```bash
SentimentAnalysis/
├── aclImdb/
│   ├── train/
│   │   ├── pos/
│   │   ├── neg/
│   │   └── unsup/
│   └── test/
│       ├── pos/
│       └── neg/
├── params/
├── src/
│   └── sentiment_vectors/
│       ├── __init__.py
│       ├── data.py
│       ├── vocabulary.py
│       ├── semantic.py
│       ├── sentiment.py
│       ├── evaluation.py
│       └── utils.py
├── Scripts/
│   ├── step1_load_data.py
│   ├── step2_build_vocab.py
│   ├── step3_train_semantic.py
│   ├── step4_train_full.py
│   └── step5_evaluate.py
├── pyproject.toml
└── README.md
```

## Dataset

The project uses the IMDb dataset introduced by Maas et al. (2011).

```bash 
aclImdb/
├── train/
│   ├── pos/    positive labeled reviews
│   ├── neg/    negative labeled reviews
│   └── unsup/  unlabeled reviews
└── test/
    ├── pos/    positive test reviews
    └── neg/    negative test reviews
```

The training data contains:
25,000 labeled reviews
50,000 unlabeled reviews
25,000 test reviews

## Example results

Nearest neighbors obtained from the learned word vectors:

| Query word | Semantic + Sentiment                              | Semantic Only                                 |
| ---------- | ------------------------------------------------- | --------------------------------------------- |
| wonderful  | perfect, fantastic, deserved, incredible, awesome | perfect, amazing, excellent, superb, fabulous |
| terrible   | awful, horrible, plain, corny, crap               | awful, horrible, atrocious, 1/10, badly       |
| boring     | terrible, predictable, awful, annoying, honestly  | amateur, pointless, mess, 1/10, terrible      |
| romantic   | romance, appealing, charm, charming, performer    | romance, charming, appealing, roles, charm    |

## Classification results:

| Method             | Our Result |  Paper |
| ------------------ | ---------: | -----: |
| Bag of Words (bnc) |     87.94% | 87.80% |
| Our Semantic Only  |     82.84% | 87.30% |
| Our Full           |     87.52% | 87.44% |
| Our Full + BoW     |     88.06% | 88.89% |

## Reference

Maas, Andrew L., Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.
Learning Word Vectors for Sentiment Analysis.
ACL 2011.
Dataset: IMDb Large Movie Review Dataset.# Sentiment-Analysis
