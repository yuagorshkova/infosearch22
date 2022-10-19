import pymorphy2
from setup_nltk import sw
from setup_nltk import word_tokenize


class Preprocessor:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = sw
        self.stop_words.add('это')

    def get_lemma(self, word: str) -> str:
        """
        :param word:
        :return: normal form of the word
        """
        return self.morph.parse(word)[0].normal_form

    def preprocess_text(self, text: str) -> str:
        """
        :param text: :return: string of preprocessed text (lemmatized, lowercased, excluding stopwords and tokens
        that include non-alphabetical characters)
        """
        words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in self.stop_words]
        return ' '.join(self.get_lemma(w) for w in words)


preprocessor = Preprocessor()
