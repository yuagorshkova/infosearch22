import numpy as np
from utils import download_and_unzip
from inverted_index import create_inverted_index_matrix, create_inverted_index_dictionary

DATA_URL = "https://github.com/yuagorshkova/infosearch22/blob/main/friends-data.zip?raw=true"

if __name__ == "__main__":
    corpus_directory = download_and_unzip(DATA_URL)

    inverted_index_matrix, vocab = create_inverted_index_matrix(corpus_directory)
    print(f"N of documents: {inverted_index_matrix.shape[0]}")
    print(f"N of words in matrix: {inverted_index_matrix.shape[1]}")

    inverted_index_dict = create_inverted_index_dictionary(corpus_directory)
    print(f"N of words in matrix: {len(inverted_index_dict)}")

    word_frequencies = [sum(v.values()) for v in inverted_index_dict.values()]
    most_freq_word_index = np.argmax(word_frequencies)
    most_freq_word = list(inverted_index_dict.keys())[most_freq_word_index]
    top_freq = word_frequencies[most_freq_word_index]
    print(f"The most frequent word in the collection is {most_freq_word} with {top_freq} occurrences")

    least_freq_word_index = np.argmin(word_frequencies)
    least_freq_word = list(inverted_index_dict.keys())[least_freq_word_index]
    lowest_freq = word_frequencies[least_freq_word_index]
    print(f"The least frequent word in the collection is {least_freq_word} with {lowest_freq} occurrences")

    words_with_zeros = np.unique(np.where(inverted_index_matrix.toarray() == 0)[1]) #words that are not included in at least some text
    words_found_in_all_docs = np.delete(np.arange(inverted_index_matrix.shape[1]), words_with_zeros)
    print("the following words are found in all docs of collection:")
    print(", ".join([vocab[word_i] for word_i in words_found_in_all_docs]))
