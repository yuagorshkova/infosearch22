import functions_for_matrix
import functions_for_dict
from utils import download_and_unzip
from inverted_index import create_inverted_index_matrix, create_inverted_index_dictionary

DATA_URL = "https://github.com/yuagorshkova/infosearch22/blob/main/friends-data.zip?raw=true"
CHARACTER_ALIAS_DICT = {
        "Monica": ["моника", "мон"],
        "Rachel": ["рейчел", "рейч"],
        "Chandler": ["чендлер", "чэндлер", "чен"],
        "Phoebe": ["фиби", "фибс"],
        "Ross": ["росс"],
        "Joey": ["джоуи", "джо"],
                            }

if __name__ == "__main__":
    corpus_directory = download_and_unzip(DATA_URL)

    inverted_index_matrix, vocab = create_inverted_index_matrix(corpus_directory)
    id_term_vocab = {v: k for k, v in vocab.items()}
    print(f"N of documents: {inverted_index_matrix.shape[0]}")

    inverted_index_dict = create_inverted_index_dictionary(corpus_directory)

    print("STATISTICS ON MATRIX", "\n")
    _ = functions_for_matrix.most_frequent_word(inverted_index_matrix, id_term_vocab)
    _ = functions_for_matrix.least_frequent_word(inverted_index_matrix, id_term_vocab)
    _ = functions_for_matrix.words_in_all_docs(inverted_index_matrix, id_term_vocab)
    _ = functions_for_matrix.most_mentioned_character(inverted_index_matrix, vocab, CHARACTER_ALIAS_DICT)

    print("STATISTICS ON DICT", "\n")
    _ = functions_for_dict.most_frequent_word(inverted_index_dict)
    _ = functions_for_dict.least_frequent_word(inverted_index_dict)
    _ = functions_for_dict.words_in_all_docs(inverted_index_dict)
    _ = functions_for_dict.most_mentioned_character(inverted_index_dict, CHARACTER_ALIAS_DICT)
