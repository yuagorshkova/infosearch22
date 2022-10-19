import torch
from transformers import AutoTokenizer
from transformers import AutoModel

TOKENIZER = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
MODEL = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_sentences_vectors(sentences, tokenizer, model):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.numpy()

