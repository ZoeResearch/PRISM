import sys
sys.path.append("../")

from Util import utils, training
from gensim.models import KeyedVectors
import os
import numpy as np
import pickle
import collections
import random
import argparse
from pre_train_def.w2v import train_word2vec
from pre_train_def.fasttext import train_fasttext
from pre_train_def.glove import train_glove, gen_corpus, gen_vocab
from Bert.utils import bert_encode, roberta_encode, get_part_embeddings
from transformers import BertTokenizer, RobertaTokenizer, TFRobertaModel, AutoTokenizer, AutoModel



def get_train_test(all_code, model, voc_size, sentence_length, mul_bin_flag, embed_name):
    train, val, test = [], [], []
    fix, unfix = [], []

    for i in range(len(all_code)):
        if all_code[i].cls == 1:
            fix.append(all_code[i])
        else:
            unfix.append(all_code[i])
    # 6:2:2
    for i in range(int(len(fix)*0.6)):
        train.append(fix[i])
        train.append(unfix[i])
    for i in range(int(len(fix)*0.6), int(len(fix)*0.8)):
        val.append(fix[i])
        val.append(unfix[i])
    for i in range(int(len(fix)*0.8), int(len(fix))):
        test.append(fix[i])
        test.append(unfix[i])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    if embed_name == "bert_seq":
        train_input_ids, train_token_type_ids, train_attention_mask, y_train = bert_encode(train, "bert-base-uncased", 200, mul_bin_flag)
        val_input_ids, val_token_type_ids, val_attention_mask, y_val = bert_encode(val, "bert-base-uncased", 200, mul_bin_flag)
        test_input_ids, test_token_type_ids, test_attention_mask, y_test = bert_encode(test, "bert-base-uncased", 200, mul_bin_flag)
        x_train = [np.asarray(train_input_ids), np.asarray(train_token_type_ids), np.asarray(train_attention_mask)]
        x_val = [np.asarray(val_input_ids), np.asarray(val_token_type_ids), np.asarray(val_attention_mask)]
        x_test = [np.asarray(test_input_ids), np.asarray(test_token_type_ids), np.asarray(test_attention_mask)]

    elif embed_name == "codebert_seq" or embed_name == "roberta_seq":
        if embed_name == "codebert_seq":
            tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        else:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        train_input_ids, train_token_type_ids, train_attention_mask, y_train = roberta_encode(train, tokenizer,
                                                                                           200, mul_bin_flag)
        val_input_ids, val_token_type_ids, val_attention_mask, y_val = roberta_encode(val, tokenizer, 200,
                                                                                   mul_bin_flag)
        test_input_ids, test_token_type_ids, test_attention_mask, y_test = roberta_encode(test, tokenizer, 200,
                                                                                       mul_bin_flag)
        x_train = [np.asarray(train_input_ids), np.asarray(train_token_type_ids), np.asarray(train_attention_mask)]
        x_val = [np.asarray(val_input_ids), np.asarray(val_token_type_ids), np.asarray(val_attention_mask)]
        x_test = [np.asarray(test_input_ids), np.asarray(test_token_type_ids), np.asarray(test_attention_mask)]

    elif embed_name == "codebert_token":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        x_train, y_train, train_length = get_part_embeddings(tokenizer, model, train, sentence_length, voc_size, mul_bin_flag)
        x_val, y_val, val_length = get_part_embeddings(tokenizer, model, val, sentence_length, voc_size, mul_bin_flag)
        x_test, y_test, test_length = get_part_embeddings(tokenizer, model, test, sentence_length, voc_size, mul_bin_flag)
        x_train, x_val, x_test = np.asarray(x_train), np.asarray(x_val), np.asarray(x_test)
    # elif embed

    else:
        x_train, y_train = get_vec_label(model, train, voc_size, sentence_length, mul_bin_flag)
        x_val, y_val = get_vec_label(model, val, voc_size, sentence_length, mul_bin_flag)
        x_test, y_test = get_vec_label(model, test, voc_size, sentence_length, mul_bin_flag)

    return x_train, np.asarray(y_train), x_val, np.asarray(y_val), x_test, np.asarray(y_test), train, val, test

def encode(bert_model_name, embed_model, all_doc, sentence_length, voc_size, mul_bin_flag):
    if bert_model_name == "codebert_token":
        model_name = "microsoft/codebert-base"
    elif bert_model_name == "roberta_token":
        model_name = "roberta-base"
    elif bert_model_name == "bert_token":
        model_name = "bert-base-uncased"
    else:
        print("no specify bert model name")
        exit(0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_vec, all_label, all_code_filter = get_part_embeddings(tokenizer, embed_model, all_doc, sentence_length, voc_size, mul_bin_flag)
    return all_vec, all_label, all_code_filter

def get_bert_train_test(all_code, all_vec, all_label):
    fix_index, unfix_index = [], []
    train_index, val_index, test_index = [], [], []
    for i in range(len(all_code)):
        if all_code[i].cls == 1:
            fix_index.append(i)
        else:
            unfix_index.append(i)

    length = min(len(fix_index), len(unfix_index))
    for i in range(int(length * 0.6)):
        train_index.append(fix_index[i])
        train_index.append(unfix_index[i])
    for i in range(int(length * 0.6), int(length * 0.8)):
        val_index.append(fix_index[i])
        val_index.append(unfix_index[i])
    for i in range(int(length * 0.8), int(length)):
        test_index.append(fix_index[i])
        test_index.append(unfix_index[i])

    random.shuffle(train_index)
    random.shuffle(val_index)
    random.shuffle(test_index)

    x_train = [all_vec[index] for index in train_index]
    y_train = [all_label[index] for index in train_index]
    x_val = [all_vec[index] for index in val_index]
    y_val = [all_label[index] for index in val_index]
    x_test = [all_vec[index] for index in test_index]
    y_test = [all_label[index] for index in test_index]

    train_doc, val_doc, test_doc = [all_code[index] for index in train_index], [all_code[index] for index in val_index], [all_code[index] for index in test_index]

    return np.asarray(x_train).astype(np.float16), np.asarray(y_train), np.asarray(x_val).astype(np.float16), np.asarray(y_val), np.asarray(x_test).astype(np.float16), np.asarray(y_test), train_doc, val_doc, test_doc

def count_statistic(codelist):
    num, num0, num1, num2, num3, num4, num5, num6, num7, num8, num9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in codelist:
        if 60 >= i:
            num += 1
        elif 100 >= i > 60: ##
            num0 += 1
        elif 120>=i>100:
            num1 += 1
        elif 200>=i>120:
            num2 += 1
        elif 300>=i>200:
            num3 += 1
        elif 600>=i>300:
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

def get_vec_label(model, all_code, voc_size, sentence_length, mul_bin_flag):
    all_vec_part = utils.get_vec_concat(model, all_code, voc_size, sentence_length, operator_set=None, ignore_list=None, regulate_byte_flag="False")
    all_label_part = utils.get_label(all_code, mul_bin_flag)
    # all_vec_part, all_label_part = np.asarray(all_vec_part).astype(np.float32), np.asarray(all_label_part).astype(np.float32)
    all_vec_part, all_label_part = np.asarray(all_vec_part,dtype=np.float32), np.asarray(all_label_part,dtype=np.float32)
    return all_vec_part, all_label_part

def get_test(all_code, pre_model_path, embed_arg, mul_bin_flag):
    print("loading...")
    # pre_model_path = get_path(pre_model_path, embed_arg)
    model = KeyedVectors.load(pre_model_path, mmap="r")
    x_test, y_test = get_vec_label(model, all_code, embed_arg["voc_size"], embed_arg["sentence_length"], mul_bin_flag)
    print("load finished!")
    return x_test, y_test

def top_n_precision(label, pred):
    tp, fp = 0, 0
    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1:
            tp += 1
        elif pred[i] == 1 and label[0] == 0:
            fp += 1
    print("tp:", tp)
    print("fp:", fp)
    return tp/(tp+fp)

def N_precision(ranked_label, i):
    return sum(list(ranked_label)[:i])/i

def get_rank_score(y_prob, y_label, n_bug):
    rank_result = list(zip(y_prob, y_label))
    rank_result.sort(reverse = True)
    ranked_prob, ranked_label = zip(*rank_result)
    top_n_tp = {}
    top_n_precision = {}

    for i in n_bug:
        top_n_tp[i] = sum(list(ranked_label)[:i])
        top_n_precision[i] = N_precision(ranked_label, i)
        # print("***********top" + str(i) + "***********")  #top n里fix的个数
        # print(str(i) + "_tp:", sum(list(ranked_label)[:i]))
        # # print("precision:", top_n_precision(list(ranked_label)[:i], list(ranked_pred)[:i]))
        # print(str(i) + "_precision:", N_precision(ranked_label, i))
    return top_n_tp, top_n_precision

def get_average(precision, recall, accuracy, f1, top_n_tp, top_n_precision, num, n_bug):
    print("average:")
    print("precision:", precision/num)
    print("recall:", recall/num)
    print("accuracy:", accuracy/num)
    print("f1:", f1/num)
    ave_tp, ave_precision = {}, {}
    for n in n_bug:
        temp_tp, temp_precision = 0, 0
        for i in range(num):
            temp_tp += top_n_tp[i][n]
            temp_precision += top_n_precision[i][n]
        ave_tp[n] = temp_tp/num
        ave_precision[n]  =temp_precision/num
    print("top_n_tp:", ave_tp)
    print("top_n_precision:", ave_precision)

def check(base):
    for project in os.listdir(base):
        if project != "merge":
            all = base + project + "/Findbugs-violations_" + project
            fix = base + project + "/fixed_Findbugs-violations_" + project
            unfix = base + project + "/unfixed_Findbugs-violations_" + project
            if len(pickle.load(open(all, "rb"))) == len(pickle.load(open(fix, "rb"))) + len(pickle.load(open(unfix, "rb"))):
                print("pass check")
            else:
                exit(0)

def merge_data(base, output):
    merge_doc = []
    for project in os.listdir(base):
        if project != "merge":
            project_path = base + project + "/Findbugs-violations_" + project
            merge_doc += pickle.load(open(project_path, "rb"))
    pickle.dump(merge_doc, open(output, "ab"))

def get_path(pre_model_path, embed_arg, code):
    return pre_model_path + code + "_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size']) + ".wordvectors"

def train_embed(embed, model_path, doc_path, embed_arg):
    if not embed:
        return None
    if not os.path.isdir(model_path+"/"+embed):
        os.mkdir(model_path+"/"+embed)
    if embed == "w2v":
        pre_model_path = model_path+"/"+embed+"/src_5_5_100.wordvectors"
    elif embed == "glove":
        pre_model_path = model_path+"/"+embed+"/src_15_15_50.txt"
    elif embed == "fasttext":
        pre_model_path = model_path+"/"+embed+"/src_5_5_100.wordvectors"
    elif embed == "elmo":
        pre_model_path = model_path+"/"+embed+"/src_ckpt/model.pickle"
    if not os.path.exists(pre_model_path):
        all_blocks = []
        print("training...")
        for doc in pickle.load(open(doc_path, "rb")):
            all_blocks.append(doc.words)
        print("all training code number:", len(all_blocks))
        if embed == "w2v":
            train_word2vec(all_blocks, embed_arg, pre_model_path)
            embed_model = KeyedVectors.load(pre_model_path, mmap="r")
        elif embed == "glove":
            glove_base = "../pre_train_def/GloVe-master/"
            corpus_path, vocab_path = model_path+"/"+embed+"/src_corpus.txt", model_path+"/"+embed+"/src_vocab.txt"
            gen_corpus(all_blocks, corpus_path)
            gen_vocab(all_blocks, vocab_path)
            args = embed_arg
            args["corpus"], args["vocab_file"] = corpus_path, vocab_path
            args["cooccurrence_file"], args["cooccurrence_shuf_file"], args["build_dir"], args["save_file"] = \
                glove_base+"cooccurrence.bin",glove_base+"cooccurrence.shuf.bin",glove_base+"build", glove_base+"app/vectors"
            args["glove_file"], args["w2v_file"] = glove_base+"app/vectors.txt", pre_model_path
            train_glove(args)
            embed_model = KeyedVectors.load_word2vec_format(pre_model_path)
        elif embed == "fasttext":
            train_fasttext(all_blocks, embed_arg, pre_model_path)
            embed_model = KeyedVectors.load(pre_model_path, mmap="r")
        elif embed == "elmo":
            embed_model = pickle.load(open(pre_model_path, "rb"))
    else:
        embed_model = load_embed_model(embed, model_path)
    return embed_model

def load_embed_model(embed, base):
    model_path = base+embed+"/"
    if embed == "w2v":
        pre_model_path = model_path+"src_5_5_100.wordvectors"
        return KeyedVectors.load(pre_model_path, mmap="r")
    elif embed == "glove":
        pre_model_path = model_path+"src_15_15_50.txt"
        return KeyedVectors.load_word2vec_format(pre_model_path)
    elif embed == "fasttext":
        pre_model_path = model_path+"src_5_5_100.wordvectors"
        return KeyedVectors.load(pre_model_path, mmap="r")
    elif embed == "elmo":
        pre_model_path = model_path+"src_ckpt/model.pickle"
        return pickle.load(open(pre_model_path, "rb"))

def load_embed_arg(embed):
    if embed == "w2v":
        iter_range = [5]
        window_range = [5]
        sg_range = [0]
        min_count_range = [0]
        voc_size_range = [100]
        negative_range = [5]
        sample_range = [1e-3]
        hs_range = [0]
        sentence_length_range = [200]
        # sentence_length_range = [100]
        embed_arg = dict(min_count=min_count_range[0], voc_size=voc_size_range[0], sg=sg_range[0],
                         negative=negative_range[0], sample=sample_range[0], hs=hs_range[0],
                         iter=iter_range[0], window=window_range[0], sentence_length=sentence_length_range[0])
    elif embed == "glove":
        memory = [4.0]
        num_threads = [8]
        verbose = [2]
        binary = [2]
        vocab_min_count = [2]
        vector_size = [50]
        max_iter = [15]
        window_size = [15]
        x_max = [10]
        sentence_length_range = [200]
        embed_arg = dict(memory=memory[0], num_threads=num_threads[0], verbose=verbose[0], binary=binary[0],
                         vocab_min_count=vocab_min_count[0], voc_size=vector_size[0], max_iter=max_iter[0],
                         window_size=window_size[0], x_max=x_max[0], sentence_length=sentence_length_range[0])
    elif embed == "fasttext":
        iter_range = [5]
        window_range = [5]
        sg_range = [0]
        min_count_range = [0]
        voc_size_range = [100]
        negative_range = [5]
        sample_range = [1e-3]
        hs_range = [0]
        sentence_length_range = [200]
        embed_arg = dict(min_count=min_count_range[0], voc_size=voc_size_range[0], sg=sg_range[0],
                         negative=negative_range[0], sample=sample_range[0], hs=hs_range[0],
                         iter=iter_range[0], window=window_range[0], sentence_length=sentence_length_range[0])
    elif embed == "elmo":
        prodim_range = [128 * 2]
        dropout_range = [0.1]
        n_epochs_range = [10]
        batchsize_range = [128]
        sentence_length_range = [200]
        embed_arg = dict(voc_size=prodim_range[0], dropout_range=dropout_range[0], iter=n_epochs_range[0], batchsize_range=batchsize_range[0],
                         sentence_length=sentence_length_range[0])
    elif embed == "bert_seq":
        model_name_range = ["bert-base-uncased"]
        max_length_range = [200]  # encode plus data have start and end
        embed_arg = dict(model_name=model_name_range[0], max_length=max_length_range[0], voc_size=0, sentence_length=0)
    elif embed == "codebert_seq":
        model_name_range = ["microsoft/codebert-base"]
        max_length_range = [200]  # encode plus data have start and end
        embed_arg = dict(model_name=model_name_range[0], max_length=max_length_range[0], voc_size=0, sentence_length=0)
    elif embed == "codebert_token":
        model_name_range = ["microsoft/codebert-base"]
        embed_arg = dict(model_name=model_name_range[0], voc_size=768, sentence_length=200)
    elif embed == "bert_token":
        model_name_range = ["bert-base-uncased"]
        embed_arg = dict(model_name=model_name_range[0], voc_size=768, sentence_length=200)
    elif embed == "roberta_token":
        model_name_range = ["roberta-base"]
        embed_arg = dict(model_name=model_name_range[0], voc_size=768, sentence_length=200)
    elif embed == "roberta_seq":
        model_name_range = ["roberta-base"]
        max_length_range = [200]  # encode plus data have start and end
        embed_arg = dict(model_name=model_name_range[0], max_length=max_length_range[0], voc_size=0, sentence_length=0)
    else:
        embed_arg = {}

    return embed_arg


def rank_pattern(dataset, keyword):
    pattern_dic = {}
    for item in dataset:
        pattern = item.info[keyword]
        if pattern not in pattern_dic.keys():
            pattern_dic[pattern] = 1
        else:
            pattern_dic[pattern] += 1
    return sorted(pattern_dic.items(), key = lambda x: x[1], reverse=True)
