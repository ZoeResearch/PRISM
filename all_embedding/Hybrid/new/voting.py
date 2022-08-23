import pickle

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
import sys
sys.path.append("../../")
from app.appUtils import load_embed_model,get_train_test, get_rank_score, load_embed_arg
from app.sort_bug import prepare_data, train
from Util.utils import get_score_binaryclassfication, write_record
from Util.training import cherry_pick_new
import tensorflow as tf
import collections
import copy
import os
import argparse
import shutil
import csv

CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')
def hard_vote(pred_results, x_test, y_test):
    pred_result_final = []
    for i in range(len(x_test)):
        temp = 0
        for j in range(len(pred_results)):
            temp += pred_results[j][i]
        if temp > len(pred_results) / 2:
            pred_result_final.append(1)
        else:
            pred_result_final.append(0)
    hard_vote_test_score = get_score_binaryclassfication(pred_result_final, y_test)
    return hard_vote_test_score

def soft_vote(pred_prob_results, x_test, y_test):
    pred_result_mean = []
    pred_result_prob = []
    # weight = [0.05, 0.1, 0.15, 0.3, 0.4]
    # weight = [0.5, 0.2, 0.15, 0.1, 0.05]
    for i in range(len(x_test)):
        # temp = np.asarray([0, 0])
        temp = 0
        for j in range(len(pred_prob_results)):
            temp += pred_prob_results[j][i][0]
        score = temp/len(pred_prob_results)
        # for j in range(len(pred_prob_results)):
        #     temp += weight[j]*pred_prob_results[j][i][0]
        # score = temp
        pred_result_prob.append(score)
        # pred_result_mean.append(np.argmax(temp / len(pred_prob_results), axis=-1))
        if score > 0.5:
            pred_result_mean.append(1)
        else:
            pred_result_mean.append(0)
    soft_vote_test_score = get_score_binaryclassfication(pred_result_mean, y_test)
    return soft_vote_test_score, pred_result_prob

def walk(num):
    global results
    if len(num) == 0:
        return
    if num not in results:
        results.append(num)

    for i in range(len(num)):
        temp = copy.deepcopy(num)
        temp.pop(i)
        walk(temp)

def pick_para(embed_name):
    if embed_name in ["w2v", "fasttext"]:
        voc_size, sentence_length = 100, 200
    elif embed_name == "glove":
        voc_size, sentence_length = 50, 200
    elif embed_name == "elmo":
        voc_size, sentence_length = 256, 200
    else:
        voc_size, sentence_length = 0, 0
        print("no embed_name!")
        exit(0)
    return voc_size, sentence_length

def load_detect_arg(detect_base_ori, embed_name, detect_name):
    result_path = detect_base_ori+embed_name+"_"+detect_name+"/best_score_record"
    detect_arg = {}
    map1 = dict(batch_size_range="batch_size", epochs_d_range="epochs_d",
                      lstm_unit_range="lstm_unit", optimizer_range="optimizer",
                      layer_range="layer", drop_out_range="drop_out",
                      learning_rate_range="learning_rate", gru_unit_range="gru_unit",
                      dense_unit_range="dense_unit", pool_size_range="pool_size",
                      kernel_size_range="kernel_size")
    map2 = {}
    with open(result_path, "r") as f:
        rows = f.readlines()
        titles = rows[0].split("\t")
        values = rows[1].split("\t")
        for i in range(len(titles)):
            if titles[i] == "batch_size":
                start = i
            if titles[i] == "kernel_size":
                end = i
        for j in range(start, end+1):
            map2[titles[j]] = values[j]
    for index in map1.keys():
        if index == "optimizer_range":
            detect_arg[index] = [map2[map1[index]]]
        elif index == "drop_out_range" or index == "learning_rate_range":
            detect_arg[index] = [float(map2[map1[index]])]
        else:
            detect_arg[index] = [int(map2[map1[index]])]
    return detect_arg


def train_model(all_code, mul_bin_flag, ename, embed_model_base, dname, detect_model_base, detect_model_base_ori, time, conn):
    embed_model = load_embed_model(ename, embed_model_base)
    embed_arg = load_embed_arg(ename)

    single_result_save_base = detect_model_base + ename + "_" + dname + "/"
    if not os.path.exists(single_result_save_base):
        os.mkdir(single_result_save_base)

    # if os.path.exists(single_result_save_base):
    #     shutil.rmtree(single_result_save_base)
    #     os.mkdir(single_result_save_base)
    test_data_path = single_result_save_base + "testdata_" + str(time)
    x_train, y_train, x_val, y_val, x_test, y_test, test = get_train_test(all_code, embed_model,
                                                                              embed_arg["voc_size"],
                                                                              embed_arg["sentence_length"],
                                                                              mul_bin_flag, ename)
    pickle.dump([x_test, y_test], open(test_data_path, "wb"))

    detect_arg = load_detect_arg(detect_model_base_ori, ename, dname)
    best_model, args, test_score, pred_y = cherry_pick_new(x_train, y_train, x_val, y_val,
                                                           x_test, y_test, embed_arg, detect_arg,
                                                           time, mul_bin_flag,
                                                           dname, conn, single_result_save_base,
                                                           ename)
    return

def group_vote(embed_model_base, embed_names, detect_model_base, detect_names, results, save_base, time, n_bug, embed_setting, soft_group_names, hard_group_names):
    pred_prob_results = {}
    pred_results = {}
    num_index = {}
    i = 0
    conn = "src"
    for ename in embed_names:
        # embed_model = load_embed_model(ename, embed_model_base)
        # embed_arg = load_embed_arg(ename)
        # x_train, y_train, x_val, y_val, x_test, y_test, test = get_train_test(all_code, embed_model,
        #                                                                       embed_arg["voc_size"],
        #                                                                       embed_arg["sentence_length"],
        #                                                                       mul_bin_flag, ename)
        # pickle.dump([x_test, y_test], open(test_data_path, "wb"))
        for dname in detect_names:
            single_result_save_base = detect_model_base + ename + "_" + dname + "/"
            detect_model_path = single_result_save_base + "best_model_" + str(time) + ".h5"
            test_data_path = single_result_save_base + "testdata_" + str(time)
            if os.path.exists(detect_model_path) and os.path.exists(test_data_path):
                index = ename + "_" + dname
                num_index[i] = index
                i += 1
                [x_test, y_test] = pickle.load(open(test_data_path, "rb"))
                print("processing index:", index)
                best_model = tf.keras.models.load_model(detect_model_path)
                pred_y_prob = best_model.predict(x_test)
                pred_y = (best_model.predict(x_test) > 0.5).astype("int32").flatten()
                pred_prob_results[index] = pred_y_prob
                pred_results[index] = pred_y

    # if len(results) > 1:
    #     for i in range(len(results))[::-1]:
    #         for num in results[i]:
    #             if num_index[num] not in pred_prob_results.keys():
    #                 results.pop(i)
    # assert len(pred_prob_results) == len(pred_results) == 1

    # if embed_setting == "w2v_bgru":
    #     test_score = get_score_binaryclassfication(pred_results[embed_setting], y_test)
    #     test_score["top_n_tp"], test_score["top_n_precision"] = get_rank_score(pred_prob_results[embed_setting], y_test.tolist(), n_bug)
    #
    #     write_record({"index": "['w2v_bgru']"}, test_score, save_base + "soft_vote_results")

    if "group" in embed_setting:
        pred_prob = []
        pred = []
        for name in soft_group_names:
            pred_prob.append(pred_prob_results[name])
        soft_vote_score, soft_prob = soft_vote(pred_prob, x_test, y_test)
        soft_best_score = soft_vote_score
        soft_best_index = soft_group_names
        soft_best_pred = soft_prob

        for name in hard_group_names:
            pred.append(pred_results[name])
        hard_vote_score = hard_vote(pred, x_test, y_test)
        hard_best_score = hard_vote_score
        hard_best_index = hard_group_names

    else:
        soft_best_f1, hard_best_f1 = -1, -1
        for group_num in results:
            pred_prob = []
            pred = []
            for num in group_num:
                pred_prob.append(pred_prob_results[num_index[num]])
                pred.append(pred_results[num_index[num]])
            soft_vote_score, soft_prob = soft_vote(pred_prob, x_test, y_test)
            hard_vote_score = hard_vote(pred, x_test, y_test)
            if soft_vote_score["f1"] > soft_best_f1:
                soft_best_f1 = soft_vote_score["f1"]
                soft_best_score = soft_vote_score
                soft_best_num = group_num
                soft_best_pred = soft_prob
            if hard_vote_score["f1"] > hard_best_f1:
                hard_best_f1 = hard_vote_score["f1"]
                hard_best_score = hard_vote_score
                hard_best_num = group_num
        soft_best_index = [num_index[i] for i in soft_best_num]
        hard_best_index = [num_index[i] for i in hard_best_num]

    pred_data = list(zip(x_test, y_test, soft_prob))
    pred_data = sorted(pred_data, key=lambda k: k[2], reverse=True)
    pickle.dump(pred_data, open(save_base + "testdata_sort_"+ str(time), "wb"))
    print("dump finished")
    print("best_soft_score:", soft_best_score)
    print("best_soft_index", soft_best_index)
    print("best_hard_score:", hard_best_score)
    print("best_hard_index", hard_best_index)
    # add rank
    soft_best_score["top_n_tp"], soft_best_score["top_n_precision"] = get_rank_score(soft_best_pred, y_test.tolist(), n_bug)
    # utils.write_record(args, test_score, save_base+"/best_score_record")
    print("one time:", soft_best_score)

    write_record({"index": soft_best_index}, soft_best_score, save_base + "soft_vote_results")
    write_record({"index": hard_best_index}, hard_best_score, save_base + "hard_vote_results")

    return

def read_csv_file(path):
    with open(path, mode='rt', encoding='UTF-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        # col = [float(row["f1"].replace("%", "")) for row in reader]
        col_f1 = [round(float(row['f1']),15) for row in reader]
        col_acc = [round(float(row['accuracy']),15) for row in reader]
    return col_f1, col_acc

def rank(detect_model_base, embed_names, detect_names):
    rank_f1 = []
    rank_acc = []
    for ename in embed_names:
        for dname in detect_names:
            single_result_save_base = detect_model_base + ename + "_" + dname + "/best_score_record.csv"
            f1, acc = read_csv_file(single_result_save_base)
            temp_f1 = (ename + "_" + dname, sum(f1)/len(f1))
            rank_f1.append(temp_f1)
            # temp_acc = (ename + "_" + dname, sum(acc) / len(acc))
            # rank_acc.append(temp_acc)
    # pred_data = list(zip(x_test, y_test, soft_prob))
    pred_f1 = sorted(rank_f1, key=lambda k: k[1], reverse=True)
    # pred_acc = sorted(rank_acc, key=lambda k: k[1], reverse=True)
    return pred_f1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', '-e')
    parser.add_argument('--vote', '-v')
    # parser.add_argument('--detect', '-d')
    args = parser.parse_args()
    embed_model_base = "../../app/model/src_top_20/"
    detect_model_base = "../../app/score/"

    violation_doc_path = "../../app/data/src_top_20"
    vote_save_base = "../../app/vote_score/vote_"+args.embed+"/"

    embed_names = ["w2v", "fasttext", "glove", "elmo"]
    detect_names = ["gru", "bgru", "lstm", "blstm", "textcnn"]
    # first = rank(detect_model_base_ori, embed_names, detect_names)
    # second = rank(detect_model_base, embed_names, detect_names)

    mul_bin_flag = 0
    filter_flag = 0
    balance_flag = 1
    n_bug = [5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 1500, 2000]
    if args.vote == "True":
        if args.embed == "all":
            results = [list(range(12))]
            embed_names = ["w2v", "fasttext", "glove", "elmo"]
            detect_names = ["bgru", "blstm", "textcnn"]
            soft_group_name, hard_group_name = [], []
        elif args.embed == "w2v_bgru":
            results = []
            embed_names = ["w2v"]
            detect_names = ["bgru"]
            soft_group_name = ['w2v_bgru']
            hard_group_name = ['w2v_bgru']

        elif args.embed == "group3":
            results = []
            embed_names = ["w2v", "fasttext"]
            detect_names = ["gru", "bgru", "lstm", "blstm", "textcnn"]
            soft_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru']
            hard_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru']

        elif args.embed == "group5":
            results = []
            embed_names = ["w2v", "fasttext", "elmo"]
            detect_names = ["gru", "bgru", "lstm", "blstm", "textcnn"]
            soft_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn']
            hard_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn']

        elif args.embed == "group7":
            results = []
            embed_names = ["w2v", "fasttext", "glove", "elmo"]
            detect_names = ["gru", "bgru", "lstm", "blstm", "textcnn"]
            soft_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm', 'fasttext_blstm']
            hard_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm', 'fasttext_blstm']

        elif args.embed == "group9":
            results = []
            embed_names = ["w2v", "fasttext", "glove", "elmo"]
            detect_names = ["gru", "bgru", "lstm", "blstm", "textcnn"]
            soft_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm', 'fasttext_blstm', 'glove_bgru', 'fasttext_textcnn']
            hard_group_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm', 'fasttext_blstm', 'glove_bgru', 'fasttext_textcnn']

        if not os.path.exists(vote_save_base):
            os.mkdir(vote_save_base)
        for i in range(1):
            group_vote(embed_model_base, embed_names, detect_model_base, detect_names, results,
                     vote_save_base, i, n_bug, args.embed, soft_group_name, hard_group_name)
    # else:
    #     if not os.path.exists(detect_model_base):
    #         os.mkdir(detect_model_base)
    #     for i in range(10):
    #         print("*********************time: "+str(i)+"************************")
    #         all_code, test = prepare_data(violation_doc_path, filter_flag, balance_flag, generate_test_flag=0)
    #         train_model(all_code, mul_bin_flag, args.embed, embed_model_base, args.detect, detect_model_base, detect_model_base_ori,
    #                         i, "src")
            # group_vote(embed_model_base, embed_names, detect_model_base, detect_model_base_ori, detect_names, results, mul_bin_flag, all_code, vote_save_base, i, n_bug)

