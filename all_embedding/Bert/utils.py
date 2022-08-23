import argparse
import collections
import tensorflow as tf
from transformers import BertTokenizer, RobertaTokenizer, RobertaModel, AutoModelForSequenceClassification, TrainingArguments
from tensorflow.python import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from transformers import TFBertModel
from Util.utils import split_codedoc
import gc
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def convert_example_to_feature(review, tokenizer, max_length):
    # max_length = 200
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length = True,  # add [PAD] tokens  padding='longest'
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True
                                 )


def encode_examples(code_docs, name, max_length, mul_bin_flag):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    tokenizer = BertTokenizer.from_pretrained(name)
    for i in range(len(code_docs)):
        bert_input = convert_example_to_feature(code_docs[i].words, tokenizer, max_length)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        if mul_bin_flag == 1:
            label_list.append(code_docs[i].cls)
        elif mul_bin_flag == 0:
            if code_docs[i].cls != 0:
                label_list.append(1)
            else:
                label_list.append(0)
    print("length of bert_input", len(input_ids_list))
    return input_ids_list, token_type_ids_list, attention_mask_list, label_list

def prepare_bert_data(doc_path, rank_base, k, mul_bin_flag, model_name, max_length):
    all_input_ids = {}
    all_token_type_ids = {}
    all_attention_mask = {}
    all_label_part = {}
    all_codedoc = split_codedoc(doc_path, rank_base, k, mul_bin_flag)
    for i in range(k):
        all_input_ids[i], all_token_type_ids[i], all_attention_mask[i], all_label_part[i] = encode_examples(all_codedoc[i], model_name, max_length, mul_bin_flag)
    return all_input_ids, all_token_type_ids, all_attention_mask, all_label_part

def get_part_embeddings(tokenizer, model, code_docs, sentence_length, voc_size, mul_bin_flag):
    embeddings = []
    length = []
    label_list = []
    for i in range(len(code_docs)):
        if mul_bin_flag == 1:
            label_list.append(code_docs[i].cls)
        elif mul_bin_flag == 0:
            if code_docs[i].cls != 0:
                label_list.append(1)
            else:
                label_list.append(0)
        doc = code_docs[i]
        if doc.words:
            tokens = tokenizer.tokenize(doc.words)
            split_tokens = [tokenizer.cls_token] + tokens
            tokens_ids = tokenizer.convert_tokens_to_ids(split_tokens)
            context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
            context_embeddings = np.asarray(context_embeddings).reshape(-1, 768)[1:]
            length.append(len(context_embeddings))
            if len(context_embeddings) > sentence_length:
                context_embeddings = context_embeddings[:sentence_length]
            else:
                temp = []
                for i in range(sentence_length-len(context_embeddings)):
                    temp.append([0]*voc_size)
                context_embeddings = np.vstack([context_embeddings, np.asarray(temp)])

        embeddings.append(context_embeddings)

    return embeddings, label_list, length

def get_codebert_word_input(doc_path, rank_base, k, mul_bin_flag, tokenizer, model, sentence_length, voc_size):
    all_codedoc = split_codedoc(doc_path, rank_base, k, mul_bin_flag)
    all_word_embeddings = []
    all_length = []
    all_label_list = []
    for i in range(k):
        embeddings, labels, length = get_part_embeddings(tokenizer, model, all_codedoc[i], sentence_length, voc_size, mul_bin_flag)
        all_word_embeddings.append(embeddings)
        all_length += length
        all_label_list.append(labels)
    return all_word_embeddings, all_label_list

def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)

def get_hidden_states(encoded, token_ids_word, model, layers, model2):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded = encoded.to(device)
    with torch.no_grad():
        output = model(**encoded)
        output2 = model2.bert(**encoded)

    # Get all hidden states
    states = output.hidden_states
    states2 = output2.hidden_states
    # print(states2.tolist==states2.tolist())
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)

def get_word_vector(sent, idx, tokenizer, model, layers, model2):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    # encode.word_ids 将每个tokenize后的token对应值的列表
    return get_hidden_states(encoded, token_ids_word, model, layers, model2)

def generate_bert_word_embeddings(model, tokenizer, doc_path):
    # Use last four layers by default
    layers = [-4, -3, -2, -1]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    training_args = TrainingArguments("test_trainer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    model2.to(device)
    word_embeddings = []
    sent = "I like cookies and I love swimming . But sometimes I getsome ."
    # idx = get_word_idx(sent, "I")
    for idx in range(len(sent.split(" "))):
        word_embedding = get_word_vector(sent, idx, tokenizer, model, layers, model2)
        word_embeddings.append(word_embedding)
    return word_embeddings

