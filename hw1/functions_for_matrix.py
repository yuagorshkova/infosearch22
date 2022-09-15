import numpy as np
from collections import Counter


# Сейчас я бы лучше сделала класс обратного индекса и все это его методами
def most_frequent_word(inverted_index_matrix, id_term_vocab):
    most_freq_index = inverted_index_matrix.sum(axis=0).argmax()
    most_freq_word = id_term_vocab[most_freq_index]
    top_freq = inverted_index_matrix[:, most_freq_index].sum()
    print(f"The most frequent word in the collection is {most_freq_word} with {top_freq} occurrences", end="\n")
    return most_freq_word


def least_frequent_word(inverted_index_matrix, id_term_vocab):
    least_freq_index = inverted_index_matrix.sum(axis=0).argmin()
    least_freq_word = id_term_vocab[least_freq_index]
    lowest_freq = inverted_index_matrix[:, least_freq_index].sum()
    print(f"The least frequent word in the collection is {least_freq_word} with {lowest_freq} occurrences", end="\n")
    return least_freq_word


def words_in_all_docs(inverted_index_matrix, id_term_vocab):
    words_with_zeros = np.unique(
        np.where(inverted_index_matrix.toarray() == 0)[1])  # words that are not included in at least some text
    words_found_in_all_docs_indices = np.delete(np.arange(inverted_index_matrix.shape[1]), words_with_zeros)
    words_found_in_all_docs = [id_term_vocab[word_i] for word_i in words_found_in_all_docs_indices]
    print("the following words are found in all docs of collection:")
    print(", ".join(words_found_in_all_docs), end="\n")

    return words_found_in_all_docs


def most_mentioned_character(inverted_index_matrix, vocab, character_alias_dict):
    character_mention_frequencies = Counter()
    for character in character_alias_dict:
        aliases_indices = [vocab[a] for a in character_alias_dict[character] if a in vocab]
        character_mention_frequencies[character] = inverted_index_matrix[:, aliases_indices].sum()
    print(f"The most frequently mentioned character is {character_mention_frequencies.most_common(1)[0]}")
    print("Character mentions range as follows:")
    print(character_mention_frequencies.most_common(), end="\n")
    return character_mention_frequencies
