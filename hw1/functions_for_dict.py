import numpy as np
from collections import Counter


def most_frequent_word(inverted_index_dict):
    word_frequencies = [sum(v.values()) for v in inverted_index_dict.values()]
    most_freq_word_index = np.argmax(word_frequencies)
    most_freq_word = list(inverted_index_dict.keys())[most_freq_word_index]
    top_freq = word_frequencies[most_freq_word_index]
    print(f"The most frequent word in the collection is {most_freq_word} with {top_freq} occurrences", end="\n")
    return most_freq_word


def least_frequent_word(inverted_index_dict):
    word_frequencies = [sum(v.values()) for v in inverted_index_dict.values()]
    least_freq_word_index = np.argmin(word_frequencies)
    least_freq_word = list(inverted_index_dict.keys())[least_freq_word_index]
    lowest_freq = word_frequencies[least_freq_word_index]
    print(f"The least frequent word in the collection is {least_freq_word} with {lowest_freq} occurrences", end="\n")
    return least_freq_word


# допустим, мы его не знаем изначально
def get_n_docs(inverted_index_dict):
    doc_indices = set()
    for v in inverted_index_dict.values():
        doc_indices.update(v)
    return len(doc_indices)


def words_in_all_docs(inverted_index_dict):
    n_docs = get_n_docs(inverted_index_dict)
    words = [k for k, v in inverted_index_dict if len(v) == n_docs]
    print("the following words are found in all docs of collection:")
    print(", ".join(words), end="\n")
    return words


def most_mentioned_character(inverted_index_dict, character_alias_dict):
    character_mention_frequencies = Counter()
    for character in character_alias_dict:
        character_mention_frequencies[character] = sum([sum(inverted_index_dict[alias].values())
                                                        for alias in character_alias_dict[character]])
    print(f"The most frequently mentioned character is {character_mention_frequencies.most_common(1)[0]}")
    print("Character mentions range as follows:")
    print(character_mention_frequencies.most_common(), end="\n")
    return character_mention_frequencies
