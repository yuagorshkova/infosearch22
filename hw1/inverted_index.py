import os
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import Preprocessor
from collections import defaultdict, Counter


def create_corpus(directory: str) -> List[str]:
    prep = Preprocessor()
    corpus = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as infile:
                if not filename == ".DS_Store":
                    episode_text = infile.read()
                    corpus.append(prep.preprocess_text(episode_text))
    return corpus


def create_inverted_index_matrix(directory: str):
    corpus = create_corpus(directory)
    vectorizer = CountVectorizer()
    inverted_index = vectorizer.fit_transform(corpus)
    vocab = vectorizer.vocabulary_
    return inverted_index, vocab


def create_inverted_index_dictionary(directory: str) -> dict:
    corpus = create_corpus(directory)
    inverted_index = defaultdict(Counter)
    for i, doc in enumerate(corpus):
        for word in doc.split():
            inverted_index[word].update([i])
    return inverted_index
