import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
import pickle
import random

import numpy as np
from transformers import TFBertModel, TFRobertaModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=config)
# tf.compat.v1.disable_eager_execution()
# tf.config.optimizer.set_jit(True)
import sys
sys.path.append("../")
from Util import utils, training
from app.HWP import HWP
from app.appUtils import *
from Util.utils import get_score_binaryclassfication, get_score_multiclassfication, get_vec_concat, write_record
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from Bert import Bert_model

# CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def generate_test_data(test_flag, random_list, unfix, unfix_length):
    test = []
    if test_flag == 1:
        for i in range(unfix_length):
            if i not in random_list:
                test.append(unfix[i])
    return test

def prepare_data(doc_path, filter_flag, balance_flag, time, generate_test_flag):
    random_path = "./data/" + str(time)
    all_code = pickle.load(open(doc_path, "rb"))
    test_data = []
    if filter_flag == 1:
        temp = []
        for doc in all_code:
            if doc.info["violation_is_single_line"] == 1:
                temp.append(doc)
        all_code = temp
    if balance_flag == 1:
        temp = []
        fix = [doc for doc in all_code if doc.cls == 1]
        unfix = [doc for doc in all_code if doc.cls == 0]
        if not os.path.exists(random_path):
            with open(random_path, "w") as f:
                random_list = random.sample(range(max(len(fix), len(unfix))), min(len(fix), len(unfix)))
                f.write(" ".join([str(item) for item in random_list]))
        else:
            with open(random_path, "r") as f:
                random_list = [int(item) for item in f.read().strip().split(" ")]

        if len(fix)<len(unfix):
            temp.extend(fix)
            for i in random_list:
                temp.append(unfix[i])
            test_data = generate_test_data(generate_test_flag, random_list, unfix, len(unfix))
        else:
            temp.extend(unfix)
            for i in random_list:
                temp.append(fix[i])

        all_code = temp
        # all_code = temp[:100] + temp[len(fix):len(fix)+100]

    return all_code, test_data

def sort_test_data_pattern(prob, all_test_data):
    rank_result = list(zip(prob, all_test_data))
    rank_result.sort(reverse=True)
    ranked_data = [item[1] for item in rank_result]

    # original_sort_pattern = rank_pattern(all_test_data, "violation_type")
    part_sort_pattern = []
    for number in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000, 6000, 8000, 10000, len(all_test_data)]:
        sort_pattern = rank_pattern(ranked_data[:number], "violation_type")
        part_sort_pattern.append(sort_pattern)

    return part_sort_pattern, ranked_data


def train(all_code, embed_model, embed_arg, detect_arg, mul_bin_flag, times, classification_model, conn, save_base, embed_name):
    save_data_path = save_base + "/data_" + str(times)    # model save path
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
    if embed_name == "codebert_token" or embed_name == "bert_token" or embed_name == "roberta_token":
        sentence_length = 200
        voc_size = 768
        if not os.path.exists(save_base+"./input_of_"+ embed_name +".pkl"):
            all_vec, all_label, all_code_filter = encode(embed_name, embed_model, all_code, sentence_length, voc_size, mul_bin_flag)
            pickle.dump([all_code_filter, all_vec, all_label], open(save_base+"./input_of_"+ embed_name +".pkl", "wb"))
        else:
            all_code_filter, all_vec, all_label = pickle.load(open(save_base+"./input_of_"+ embed_name +".pkl", "rb"))

        print("spliting dataset...")
        x_train, y_train, x_val, y_val, x_test, y_test, train_doc, val_doc, test_doc = get_bert_train_test(all_code_filter, all_vec, all_label)
        detect_model, args, test_score, y_pred, pred_prob, best_model = training.cherry_pick_new(x_train, y_train,
                                                                                                     x_val, y_val,
                                                                                                     x_test, y_test,
                                                                                                     embed_arg,
                                                                                                     detect_arg, times,
                                                                                                     mul_bin_flag,
                                                                                                     classification_model,
                                                                                                     conn, save_base,
                                                                                                     embed_name)

        best_model.save_weights(save_data_path + "/best_model_" + str(times))

    else:
        x_train, y_train, x_val, y_val, x_test, y_test, train_doc, val_doc, test_doc = get_train_test(all_code, embed_model, embed_arg["voc_size"],
                                                                    embed_arg["sentence_length"], mul_bin_flag, embed_name)

        detect_model, args, test_score, y_pred, pred_prob, best_model = training.cherry_pick_new(x_train, y_train, x_val, y_val, x_test, y_test, embed_arg, detect_arg, times, mul_bin_flag, classification_model, conn, save_base, embed_name)

        if embed_name == "bert_seq" or embed_name == "codebert_seq" or embed_name == "roberta_seq":
            best_model.save_weights(save_data_path + "/best_model_" + str(times))
        else:
            best_model.save(save_data_path + "/best_model_" + str(times) + ".h5")
        # pickle.dump(train_doc, open(save_data_path+"/train_doc.pkl", "wb"))
        # pickle.dump(val_doc, open(save_data_path+"/val_doc.pkl", "wb"))
        # pickle.dump(test_doc, open(save_data_path+"/test_doc.pkl", "wb"))

    return x_test, y_test, detect_model, args, test_score, y_pred, pred_prob

def code_embedding(train_flag, code, embed_arg, detect_arg, all_code, mul_bin_flag,
                   n_bug, times, embed, classification_model, save_base, embed_name, detect_name):
    if train_flag == 1:
        # embedding_model_path = "./model/" + code + "_top_20/" + embedding + "_" + str(embed_arg['iter']) + "_" + str(
        #     embed_arg['window']) + "_" + str(embed_arg['voc_size']) + "/"
        x_test, y_test, best_model, args, test_score, y_pred, y_prob = train(all_code, embed, embed_arg, detect_arg,
                                             mul_bin_flag, times, classification_model, code, save_base, embed_name)

    else:
        # test_score = {}
        args = {}
        base = "./model/src_top_20/"
        # embedding_modl_path = "./model/src_top_20/w2v/src_5_5_100.wordvectors"
        # detect_model_path = "./score/w2v_gru/best_model_"+str(times)+".h5"
        detect_model_path = "./score/"+embed_name+"_"+detect_name+"/best_model_"+str(times)+".h5"
        if not os.path.isdir(detect_model_path):
            detect_model_path = "./score/"+embed_name+"_"+detect_name+"/best_model_0.h5"
        best_model = tf.keras.models.load_model(detect_model_path)
        # embed_model = KeyedVectors.load(embedding_model_path, mmap="r")

        embed_model = load_embed_model(embed_name, base)
        x_train, y_train, x_val, y_val, x_test, y_test = get_train_test(all_code, embed_model, embed_arg["voc_size"],
                                                                        embed_arg["sentence_length"], mul_bin_flag,
                                                                        embed_name)
        if mul_bin_flag:
            test_y_pre = np.argmax(best_model.predict(x_test), axis=-1)
            test_score = get_score_multiclassfication(test_y_pre, y_test, args["class_num"])
        else:
            test_y_pre = (best_model.predict(x_test) > 0.5).astype("int32").flatten()
            test_score = get_score_binaryclassfication(test_y_pre, y_test)


    test_score["top_n_tp"], test_score["top_n_precision"] = get_rank_score(y_prob, y_test.tolist(), n_bug)
    print("one time:", test_score)

    return test_score["precision"], test_score["recall"], test_score["accuracy"], test_score["f1"], test_score["top_n_tp"], test_score["top_n_precision"]

def gen_test_add(test_data_add, embed_model, embed_arg, mul_bin_flag):
    x_test_add, y_test_add = get_vec_label(embed_model, test_data_add, embed_arg["voc_size"], embed_arg["sentence_length"], mul_bin_flag)
    return x_test_add, np.asarray(y_test_add)

def get_score(best_model, x_test_part, y_test_part):
    test_y_pre = (best_model.predict(np.asarray(x_test_part)) > 0.5).astype("int32").flatten()
    test_score = get_score_binaryclassfication(test_y_pre, y_test_part)
    return test_score, test_y_pre

def divide_pattern(all_data, embed_model, voc_size, sentence_length):
    data_by_pattern = {}
    label_by_pattern = {}

    for item in all_data:
        pattern = item.info["violation_type"]
        cls = item.cls
        code = item.words
        code_vec = get_vec_concat(embed_model, [item], voc_size, sentence_length, operator_set=None, ignore_list=None, regulate_byte_flag="False")[0]

        if pattern not in data_by_pattern.keys():
            data_by_pattern[pattern] = []
            label_by_pattern[pattern] = []
            data_by_pattern[pattern].append(code_vec)
            label_by_pattern[pattern].append(cls)
        else:
            data_by_pattern[pattern].append(code_vec)
            label_by_pattern[pattern].append(cls)
    for pattern in data_by_pattern.keys():
        assert len(data_by_pattern[pattern]) == len(label_by_pattern[pattern])

    return data_by_pattern, label_by_pattern


def test_embedding_by_pattern(code, embed_arg, detect_arg, all_code, test_data_add, mul_bin_flag,
                   times, embed_model, classification_model, save_base, embed_name, detect_name):
    x_test, y_test, best_model, args, test_score, y_pred, all_test_data = train(all_code, embed_model, embed_arg, detect_arg,
                                                         mul_bin_flag, times, classification_model, code,
                                                         save_base, embed_name)
    x_test_add, y_test_add = gen_test_add(test_data_add, embed_model, embed_arg, mul_bin_flag)
    pickle.dump([x_test_add, y_test_add], open())
    x_test.extend(x_test_add)
    y_test.extend(y_test_add)
    result_path = "./score/by_pattern/"+embed_name+"_"+detect_name
    if os.path.exists(result_path):
        os.remove(result_path)
    test_data, test_label = divide_pattern(test_data_add+test, embed_model, embed_arg["voc_size"], embed_arg["sentence_length"])
    test_score = {}
    all_pred_label = {}
    for pattern in test_data.keys():
        test_score[pattern], predict_label = get_score(best_model, test_data[pattern], test_label[pattern])
        all_pred_label[pattern] = predict_label
        write_record({"":pattern}, test_score[pattern], result_path)
    return


def baseline_1(base, alpha, n_pattern, all_code):
    statistic_path = base + "Statistics-of-Violation-types.xlsx"
    violation_path = base + "Distribution-of-per-violation-type-in-per-project.xlsx"
    fix_violation_path = base + "Distribution-of-per-fixed-violation-type-in-per-project.xlsx"
    hwp = HWP(statistic_path, fix_violation_path, violation_path, alpha, n_pattern, all_code)
    # hwp.calc_by_pattern()
    hwp.calc_by_time()
    sample_code = random.sample(all_code, int(len(all_code)*0.4))

    # test_code =
    hwp.prioritize(sample_code)

def load_embed(embed, base):
    embed_arg = load_embed_arg(embed)
    embed_model = load_embed_model(embed, base)
    return embed_arg, embed_model

def dump_data(balanced_data, save_path):
    fix = [doc for doc in balanced_data if doc.cls == 1]
    unfix = [doc for doc in balanced_data if doc.cls == 0]
    pickle.dump(fix, open(save_path+"bad_new", "wb"))
    pickle.dump(unfix, open(save_path+"good_new", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', '-f')
    parser.add_argument('--balance', '-b')
    parser.add_argument('--train', '-t')
    parser.add_argument('--detect', '-d')
    parser.add_argument('--method', '-m')
    parser.add_argument('--embed', '-e')
    parser.add_argument('--code', '-c')
    args = parser.parse_args()
    if args.embed == "bert_seq" or args.embed == "codebert_seq" or args.embed == "roberta_seq":
        batch_size_range = [16]
        epochs_d_range = [80]
        # epochs_d_range = [1]
        optimizer_range = ["adam"]
        learning_rate_range = [1e-5, 5e-5, 1e-3]
        encode_layer_range = [0, 2, 4]
        # learning_rate_range = [0.001]

        lstm_unit_range = [0]
        layer_range = [0]
        drop_out_range = [0]
        gru_unit_range = [0]
        dense_unit_range = [0]
        pool_size_range = [0]
        kernel_size_range = [0]

    # else:
    #     batch_size_range = [32, 128]
    #     epochs_d_range = [80]
    #     optimizer_range = ["adam"]
    #     learning_rate_range = [1e-5, 5e-5, 1e-3]
    #     encode_layer_range = [0]
        # epochs_d_range = [1]

    elif args.embed == "bert_token" or args.embed == "codebert_token" or args.embed == "roberta_token":
        # lstm_unit_range = [64, 128, 256]
        lstm_unit_range = [64]
        optimizer_range = ["adam"]
        # layer_range = [2, 4, 6]
        layer_range = [2]
        drop_out_range = [0.5]
        # gru_unit_range = [64, 128]
        gru_unit_range = [64]
        # dense_unit_range = [128, 256]
        dense_unit_range = [128]
        # pool_size_range = [10,30]
        pool_size_range = [10]
        # kernel_size_range = [5,10]
        kernel_size_range = [5]
        batch_size_range = [16]
        epochs_d_range = [60]
        # learning_rate_range = [3e-5, 5e-5, 1e-3]
        learning_rate_range = [1e-5]

        # lstm_unit_range = [128]
        # optimizer_range = ["adam"]
        # layer_range = [6]
        # drop_out_range = [0.5]
        # gru_unit_range = [128]
        # dense_unit_range = [128]
        # pool_size_range = [30]
        # kernel_size_range = [10]
        # batch_size_range = [16]
        # epochs_d_range = [1]
        # learning_rate_range = [3e-5]

        encode_layer_range = [0]

    elif args.embed == "w2v":
        # lstm_unit_range = [256, 512, 1024]
        # optimizer_range = ["adam"]
        # layer_range = [6, 10, 15]
        # drop_out_range = [0.5]
        # learning_rate_range = [0.001, 0.01]
        # gru_unit_range = [128, 256]
        # dense_unit_range = [32]
        # pool_size_range = [30]
        # kernel_size_range = [10]
        # batch_size_range = [32, 64, 128]
        # epochs_d_range = [80]

        lstm_unit_range = [64]
        optimizer_range = ["adam"]
        layer_range = [8]
        drop_out_range = [0.5]
        learning_rate_range = [0.001]
        gru_unit_range = [256]
        dense_unit_range = [32]
        pool_size_range = [30]
        kernel_size_range = [10]
        batch_size_range = [32]
        epochs_d_range = [80]

        # learning_rate_range = [3e-5, 5e-5, 1e-3]
        encode_layer_range = [0]

    elif args.embed == "fasttext":
        # lstm_unit_range = [32]
        # optimizer_range = ["adam"]
        # layer_range = [4,6,8]
        # drop_out_range = [0.5]
        # learning_rate_range = [0.0003, 0.001, 0.003]
        # gru_unit_range = [256, 512]
        # dense_unit_range = [32]
        # pool_size_range = [30]
        # kernel_size_range = [10]
        # batch_size_range = [32, 64, 128]
        # epochs_d_range = [80]
        lstm_unit_range = [64]
        optimizer_range = ["adam"]
        layer_range = [4]
        drop_out_range = [0.5]
        learning_rate_range = [0.0003]
        gru_unit_range = [128]
        dense_unit_range = [32]
        pool_size_range = [30]
        kernel_size_range = [10]
        batch_size_range = [32]
        epochs_d_range = [80]

        # learning_rate_range = [3e-5, 5e-5, 1e-3]
        encode_layer_range = [0]

    elif args.embed == "glove":
        # lstm_unit_range = [64, 128, 256]
        # optimizer_range = ["adam"]
        # layer_range = [4,6,8]
        # drop_out_range = [0.5]
        # learning_rate_range = [0.001]
        # gru_unit_range = [256]
        # dense_unit_range = [32]
        # pool_size_range = [30]
        # kernel_size_range = [10]
        # batch_size_range = [32, 64, 128]
        # epochs_d_range = [80]

        lstm_unit_range = [128]
        optimizer_range = ["adam"]
        layer_range = [4]
        drop_out_range = [0.5]
        learning_rate_range = [0.001]
        gru_unit_range = [256]
        dense_unit_range = [32]
        pool_size_range = [30]
        kernel_size_range = [10]
        batch_size_range = [32]
        epochs_d_range = [80]

        # learning_rate_range = [3e-5, 5e-5, 1e-3]
        encode_layer_range = [0]

    elif args.embed == "elmo":
        lstm_unit_range = [32]
        optimizer_range = ["adam"]  
        layer_range = [4]
        drop_out_range = [0.5]
        learning_rate_range = [0.0003]
        gru_unit_range = [512]
        dense_unit_range = [32]
        pool_size_range = [30]
        kernel_size_range = [10]
        batch_size_range = [128]
        epochs_d_range = [80]

        # learning_rate_range = [3e-5, 5e-5, 1e-3]
        encode_layer_range = [0]

    else:
        batch_size_range = [32, 64, 128]
        epochs_d_range = [80]
        lstm_unit_range = [32]
        optimizer_range = ["adam"]
        layer_range = [4, 6, 8]
        drop_out_range = [0.5]
        learning_rate_range = [0.003, 0.001, 0.0003]
        gru_unit_range = [128]
        dense_unit_range = [32]
        pool_size_range = [30]
        kernel_size_range = [10]
        encode_layer_range = [0]

    detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epochs_d_range,
                      lstm_unit_range=lstm_unit_range, optimizer_range=optimizer_range,
                      layer_range=layer_range, drop_out_range=drop_out_range,
                      learning_rate_range = learning_rate_range,gru_unit_range=gru_unit_range,
                      dense_unit_range=dense_unit_range, pool_size_range=pool_size_range,
                      kernel_size_range=kernel_size_range, encode_layer_range = encode_layer_range)
    code = args.code
    violation_doc_path = "./data/src_top_20_with_time_newest_with_warning_context_for_warning_type_without_test_classes"

    if args.embed and args.detect:
        save_base = "./score/" + args.embed + "_" + args.detect+"/"
        if not os.path.isdir(save_base):
            os.mkdir(save_base)
    train_flag = int(args.train)
    filter_flag = int(args.filter)
    balance_flag = int(args.balance)
    mul_bin_flag = 0

    base = "./baseline/"
    alpha = 0.9

    if balance_flag == 1:
        num = 10
    elif balance_flag == 0:
        num = 1


    precision, recall, accuracy, f1 = 0, 0, 0, 0
    top_n_tp, top_n_precision = [], []
    top_n_pattern_precision, top_n_pattern_recall, top_n_pattern_acc, top_n_pattern_f1 = {}, {}, {}, {}
    n_bug = [5,10,20,30,40,50,100,200,500,1000,1500,2000]

    n_pattern = [5,10,20,30,40,50]
    embed_arg = load_embed_arg(args.embed)
    if args.embed == "bert_seq":
        embed_model = TFBertModel.from_pretrained("bert-base-uncased")
    # elif args.embed == "bert_token":
    elif args.embed == "codebert_seq":
        embed_model = TFRobertaModel.from_pretrained("microsoft/codebert-base")
    elif args.embed == "codebert_token":
        embed_model = AutoModel.from_pretrained("microsoft/codebert-base")
    elif args.embed == "bert_token":
        embed_model = AutoModel.from_pretrained("bert-base-uncased")
    elif args.embed == "roberta_token":
        embed_model = AutoModel.from_pretrained("roberta-base")
    elif args.embed == "roberta_seq":
        embed_model = TFRobertaModel.from_pretrained("roberta-base")
    # elif args.embed == "roberta_token":
    else:
        embed_model = train_embed(args.embed, "./model/src_top_20/", violation_doc_path, embed_arg)

    # path = save_base+"/data_0/"
    # train, val, test, pattern, ranked_doc = pickle.load(open(path+"train_doc.pkl", "rb")), \
    #                             pickle.load(open(path+"val_doc.pkl", "rb")), \
    #                             pickle.load(open(path+"test_doc.pkl", "rb")), \
    #                             pickle.load(open(path+"sort_pattern.pkl", "rb")),\
    #                             pickle.load(open(path+"ranked_test_doc.pkl", "rb"))

    for i in range(10):
    # for i in range(8, 10):
    # for i in range(1):
        all_code, test1 = prepare_data(violation_doc_path, filter_flag, balance_flag, i, generate_test_flag=0)
        if args.method == "embedding":
            single_precision, single_recall, single_accuracy, single_f1, single_top_n_tp, single_top_n_precision = \
                         code_embedding(train_flag, code, embed_arg, detect_arg, all_code,
                                        mul_bin_flag, n_bug, i, embed_model, args.detect, save_base, args.embed, args.detect)
            precision += single_precision
            recall += single_recall
            accuracy += single_accuracy
            f1 += single_f1
            top_n_tp.append(single_top_n_tp)
            top_n_precision.append(single_top_n_precision)

        elif args.method == "hwp":
            baseline_1(base, alpha, n_pattern, all_code)
    if args.method == "embedding":
        get_average(precision, recall, accuracy, f1, top_n_tp, top_n_precision, num, n_bug)
