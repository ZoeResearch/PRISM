import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
import argparse
import collections
from appUtils import *
from gensim.models import KeyedVectors
from sort_bug import get_vec_label, prepare_data, train_embed
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def dump_by_project(all_code, model, voc_size, sentence_length, mul_bin_flag, save_base, num, cls):
    # "project_name"--100  "violation_type"--50
    project_data = {}
    project_label = {}
    project_statistic = {}
    for i in range(len(all_code)):
        pro_name = all_code[i].info[cls]
        if pro_name not in project_data.keys():
            project_data[pro_name] = []
            project_data[pro_name].append(all_code[i])
        else:
            project_data[pro_name].append(all_code[i])
    for item in project_data.keys():
        project_statistic[item] = {}
        project_statistic[item]["fixed"] = 0
        project_statistic[item]["unfixed"] = 0

        for code in project_data[item]:
            if code.cls == 1:
                project_statistic[item]["fixed"] += 1
            elif code.cls == 0:
                project_statistic[item]["unfixed"] += 1

    for item in project_data.keys():
        sample_data, sample_label = [], []
        project_data[item], project_label[item] = get_vec_label(model, project_data[item], voc_size, sentence_length, mul_bin_flag)
        sample_num = min(num, len(project_data[item]))
        sample_list = random.sample(range(len(project_data[item])), sample_num)
        for i in sample_list:
            sample_data.append(project_data[item][i])
            sample_label.append(project_label[item][i])
        pickle.dump(np.asarray(sample_data), open(save_base+"/"+item+"_data.pkl", "wb"))
        pickle.dump(np.asarray(sample_label), open(save_base+"/"+item+"_label.pkl", "wb"))


def dump_by_label(all_code, sample_num, model, voc_size, sentence_length, mul_bin_flag, save_base):
    fix, unfix = [], []
    for i in range(len(all_code)):
        if all_code[i].cls == 1:
            fix.append(all_code[i])
        else:
            unfix.append(all_code[i])
    fix, fix_label = get_vec_label(model, fix, voc_size, sentence_length, mul_bin_flag)
    unfix, unfix_label = get_vec_label(model, unfix, voc_size, sentence_length, mul_bin_flag)
    for i in fix_label.tolist():
        if i != 1:
            print("error")
            exit(0)
    for i in unfix_label.tolist():
        if i != 0:
            print("error")
            exit(0)

    for i in range(50):  # sample 50 times
        fix_sample_list = random.sample(range(len(fix)), sample_num)
        unfix_sample_list = random.sample(range(len(unfix)), sample_num)
        vec_list = []
        label_list = []
        for item in fix_sample_list:
            vec_list.append(fix[item])
            label_list.append(int(fix_label[item]))
        for item in unfix_sample_list:
            vec_list.append(unfix[item])
            label_list.append(int(unfix_label[item]))
        pickle.dump(np.asarray(vec_list), open(save_base+"/src_vec_"+str(i)+".pkl", "wb"))
        pickle.dump(np.asarray(label_list), open(save_base+"/src_label_"+str(i)+".pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', '-e')
    parser.add_argument('--mul_bin_flag', '-m')
    parser.add_argument('--cls', '-c')
    args = parser.parse_args()
    violation_doc_path = "./data/src_top_20"
    pre_model_path = "./model/src_top_20/"+args.embed
    doc_path = "./data/src_top_20"
    # embed_base = "../pre_train_def/spotbugs/"
    embed_base = "./model/src_top_20/"
    # filter_flag = 1
    sample_num = 1000
    if args.embed == "w2v" or args.embed == "fasttext":
        voc_size = 100
    elif args.embed == "glove":
        voc_size = 50
    elif args.embed == "elmo":
        voc_size = 256
    sentence_length = 200
    # mul_bin_flag = int(args.mul_bin_flag)
    # if mul_bin_flag == 1:
    #     save_base = "./data/mul/"+args.embed
    #     # if os.path.isdir(save_base):
    #     #     exit(0)
    #     if not os.path.isdir("./data/mul/"):
    #         os.mkdir("./data/mul/")
    #     if not os.path.isdir(save_base):
    #         os.mkdir(save_base)
    # elif mul_bin_flag == 0:
    #     save_base = "./data/bin/"+args.embed
    #     # if os.path.isdir(save_base):
    #     #     exit(0)
    #     if not os.path.isdir("./data/bin/"):
    #         os.mkdir("./data/bin/")
    #     if not os.path.isdir(save_base):
    #         os.mkdir(save_base)

    if args.cls == "project":
        save_base = "./data/projects/"+args.embed
        if not os.path.isdir("./data/projects/"):
            os.mkdir("./data/projects/")
        if not os.path.isdir(save_base):
            os.mkdir(save_base)
    elif args.cls == "bug":
        save_base = "./data/pattern/" + args.embed
        if not os.path.isdir("./data/pattern/"):
            os.mkdir("./data/pattern/")
        if not os.path.isdir(save_base):
            os.mkdir(save_base)

    all_code = prepare_data(violation_doc_path, filter_flag=1, balance_flag=0)
    # all_balance_code = prepare_data(violation_doc_path, filter_flag=0, balance_flag=1)
    # model = train_embed(pre_model_path, doc_path)
    model = load_embed_model(args.embed, embed_base)
    # print("start dumping "+args.embed)
    # dump_by_label(all_code, sample_num, model, voc_size, sentence_length, 0, save_base)
    if args.cls == "project":
        num = 10000
        cls = "project_name"
        dump_by_project(all_code, model, voc_size, sentence_length, 0, save_base, num, cls)
    # elif args.cls == "bug":
    #     num = 2000
    #     cls = "violation_type"
    #     dump_by_project(all_code, model, voc_size, sentence_length, 0, save_base, num, cls)
