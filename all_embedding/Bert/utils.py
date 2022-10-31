import argparse
import collections
import tensorflow as tf
from transformers import BertTokenizer, RobertaTokenizer,  TFRobertaModel, RobertaModel, AutoModelForSequenceClassification, TrainingArguments
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
    # 单个数据转换为bert的输入形式 encode_plue返回三个元素的字典
    # max_length = 200
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length = True,  # add [PAD] tokens  padding='longest'
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True
                                 )





def bert_encode(code_docs, name, max_length, mul_bin_flag):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    #输入：原始代码token list， label
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

def prepare_bert_model_data(doc_path, rank_base, k, mul_bin_flag, model_name, max_length):
    all_input_ids = {}
    all_token_type_ids = {}
    all_attention_mask = {}
    all_label_part = {}
    all_codedoc = split_codedoc(doc_path, rank_base, k, mul_bin_flag)
    for i in range(k):
        all_input_ids[i], all_token_type_ids[i], all_attention_mask[i], all_label_part[i] = bert_encode(all_codedoc[i], model_name, max_length, mul_bin_flag)
    return all_input_ids, all_token_type_ids, all_attention_mask, all_label_part

def roberta_encode(code_docs, tokenizer, sentence_length, mul_bin_flag):
    MAX_LEN = sentence_length + 2
    ct = len(code_docs)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')  # Not used in text classification
    label_list = []
    for k, code_doc in enumerate(code_docs):
        # Tokenize
        text = code_doc.words
        tok_text = tokenizer.tokenize(" ".join(text))

        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN - 2)])

        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

        # Set to 1s in the attention input
        attention_mask[k, :input_length] = 1

        if mul_bin_flag == 1:
            label_list.append(code_doc.cls)
        elif mul_bin_flag == 0:
            if code_doc.cls != 0:
                label_list.append(1)
            else:
                label_list.append(0)

    return input_ids, token_type_ids, attention_mask, label_list

def prepare_roberta_model_data(doc_path, rank_base, k, mul_bin_flag, model_name, max_length, pretrain_model_name):
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_name)
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_label_part = []
    all_codedoc = split_codedoc(doc_path, rank_base, k, mul_bin_flag)
    for i in range(k):
        part_input_ids, part_token_type, part_mask, part_label = roberta_encode(all_codedoc[i], tokenizer, max_length, mul_bin_flag)
        all_input_ids.append(part_input_ids)
        all_token_type_ids.append(part_token_type)
        all_attention_mask.append(part_mask)
        all_label_part.append(part_label)
    return all_input_ids, all_token_type_ids, all_attention_mask, all_label_part

def get_part_embeddings(tokenizer, model, code_docs, sentence_length, voc_size, mul_bin_flag):
    embeddings = []
    length = []
    label_list = []
    filter_length_number = 0
    all_code = []
    for i in range(len(code_docs)):
        doc = code_docs[i]
        if doc.words:
            # print(len(doc.words))
            tokens = tokenizer.tokenize(" ".join(doc.words))
            split_tokens = [tokenizer.cls_token] + tokens
            tokens_ids = tokenizer.convert_tokens_to_ids(split_tokens)
            if len(tokens_ids) > 512:
                filter_length_number += 1
                continue

            context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
            # context_embeddings = np.asarray(context_embeddings).reshape(-1, 768)[1:]
            context_embeddings = context_embeddings.detach().numpy().reshape(-1, 768)[1:]
            length.append(len(context_embeddings))
            if len(context_embeddings) > sentence_length:
                context_embeddings = context_embeddings[:sentence_length]
            else:
                for _ in range(sentence_length-len(context_embeddings)):
                    temp=[0]*voc_size
                    context_embeddings = np.vstack([context_embeddings, np.asarray(temp)])
            embeddings.append(context_embeddings)
            all_code.append(doc)

            if mul_bin_flag == 1:
                label_list.append(code_docs[i].cls)
            elif mul_bin_flag == 0:
                if code_docs[i].cls != 0:
                    label_list.append(1)
                else:
                    label_list.append(0)

    print("filter number:", filter_length_number)
    return embeddings, label_list, all_code

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
    # count_statistic(all_length)
    return all_word_embeddings, all_label_list

def count_statistic(codelist):
    num, num0, num1, num2, num3, num4, num5, num6, num7, num8, num9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in codelist:
        if 60 >= i:
            num += 1
        elif 100 >= i > 60: ##
            num0 += 1
        elif 200>=i>100:
            num1 += 1
        elif 300>=i>200:
            num2 += 1
        elif 500>=i>300:
            num3 += 1
        elif 600>=i>500:
            num4 += 1
        elif 700>=i>600:
            num5 += 1
        elif 800>=i>700:
            num6 += 1
        elif 1000>=i>800:
            num7 += 1
        elif 1500>=i>1000:
            num8 += 1
        elif 3000>=i>1500:
            num9 += 1

    for n in [num, num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]:
        print(n)

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

