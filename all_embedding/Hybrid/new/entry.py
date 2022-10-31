import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=config)
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import gc
from PCA import PCA
import sys
sys.path.append("../../")
from Util.utils import *
from Util.training import cherry_pick, cherry_pick_new
from hybrid_utils import process_hybrid_data
from triplet_loss import triplet_loss_compare, triplet_loss_2
from Util.gen_BYTE_vec import get_byte_embedding, allIns
from Util.gen_IR_vec import get_IR_embedding
import faulthandler
faulthandler.enable()

def pca_test(all_src_vec_part, all_src_label_part, all_ir_vec_part, all_ir_label_part, all_byte_vec_part, all_byte_label_part,
             embed_arg, detect_arg, times, split_flag, K_fold, pca_k , flag, classification_model, conn, save_base):
    # assert all_src_label_part.tolist() == all_ir_label_part.tolist() == all_byte_label_part.tolist()
#     codepath = save_base + "/ir_pca.pkl"
#     if os.path.exists(codepath):
#         all_ir_vec_part = pickle.load(open(codepath, "rb"))
        # all_ir = pickle.load(open(code, "rb"))
    if conn == "pca_ir":
        all_vec, all_label = all_ir_vec_part, all_ir_label_part
    elif conn == "pca_byte":
        all_vec, all_label = all_byte_vec_part, all_byte_label_part
    for i in range(len(all_vec)):
        all_vec[i] = PCA(all_vec[i], pca_k).pca()
    for i in range(len(all_vec)):
        all_vec[i] = np.asarray(all_vec[i]).reshape(len(all_vec[i]),1,-1)
    pickle.dump(all_vec, open(save_base + "/pca.pkl", "wb"))

    cherry_pick(all_vec, all_label, embed_arg, detect_arg, times, split_flag, K_fold, flag, classification_model, conn, save_base)



def direct(src_vec_part, src_label_part, ir_vec_part, byte_vec_part, embed_arg, detect_arg, times, split_flag, k, flag, classification_model, conn, save_base):
    cherry_pick([src_vec_part, ir_vec_part, byte_vec_part], src_label_part, embed_arg, detect_arg, times, split_flag, k, flag, classification_model, conn, save_base)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_embed', '-e1')
    parser.add_argument('--src_code', '-e2')
    parser.add_argument('--ir_embed', '-i1')
    parser.add_argument('--ir_code', '-i2')
    parser.add_argument('--byte_embed', '-b1')
    parser.add_argument('--byte_code', '-b2')
    parser.add_argument('--embedding_network', "-emb")
    parser.add_argument("--retrain", "-r")
    parser.add_argument('--split_test', '-s')
    parser.add_argument('--hybrid', '-d', required=True)
    parser.add_argument('--cls', '-l', required=True)
    parser.add_argument('--conn', '-c', required=True)
    parser.add_argument('--neural', '-n', required=True)

    parser.add_argument('--cl_epoch', '-epo', required=True)
    parser.add_argument('--cl_hidden_size', '-hid', required=True)
    parser.add_argument('--cl_layers', '-lay', required=True)
    args = parser.parse_args()

    iter_range = [5]
    window_range = [5]
    sg_range = [0]
    min_count_range = [0]
    voc_size_range = [100]
    negative_range = [5]
    sample_range = [1e-3]
    hs_range = [0]
    sentence_length_range = [200]   #src==ir==byte
    embed_batch_size = 32
    embed_dim = 100
    # if args.conn == "3loss_src":
    #     embed_epoch = 60
    # else:
    #     embed_epoch = 100
    embed_epoch, hidden_size, layers = int(args.cl_epoch), int(args.cl_hidden_size), int(args.cl_layers)
    # hidden_size, layers, dropout = 64, 4, 0.5
    # hidden_size, layers, dropout = 128, 4, 0.5
    dropout = 0.5
    # batch_size_range = [32]
    # epochs_d_range = [1]
    # # epochs_d_range = [40]
    # lstm_unit_range = [32]
    # optimizer_range = ["Adam"]
    # layer_range = [2]
    # drop_out_range = [0.5]
    # learning_rate_range = [0.0003]
    # gru_unit_range = [128]
    # dense_unit_range = [32]
    # pool_size_range = [5]
    # kernel_size_range = [5]

    batch_size_range = [32, 128]
    epochs_d_range = [50]
    # epochs_d_range = [1]
    lstm_unit_range = [64, 128]
    # lstm_unit_range = [32, 64, 128]
    optimizer_range = ["Adam"]
    layer_range = [4, 6]
    drop_out_range = [0.5]
    learning_rate_range = [0.0003, 0.001, 0.002]
    gru_unit_range = [128, 256]
    dense_unit_range = [32, 64, 128]
    pool_size_range = [5, 10, 20]
    kernel_size_range = [5, 10, 20]

    times = 0
    cls = args.cls
    base = "../../"
    # base = "../"

    # doc_base = base+"pickle_object/spotbugs/detect_bin_sample_15000/"
    K_fold = 11

    if args.cls == "spot_bin":
        mul_bin_flag = 0
        class_num = 2
        doc_base = base+"pickle_object/spotbugs/detect_bin_sample_15000/"
        pre_model_base = base + "pre_train_def/spotbugs/"
        categorical_flag = "False"

    elif args.cls == "spot_mul":
        mul_bin_flag = 1
        class_num = 9
        doc_base = base + "pickle_object/spotbugs/detect_mul/"
        pre_model_base = base + "pre_train_def/spotbugs/"
        categorical_flag = "True"

    elif args.cls == "reduce_fp":
        mul_bin_flag = 0
        class_num = 2
        doc_base = base + "app/data/detect_bin/"
        pre_model_base = base + "app/model/src_top_20/"
        categorical_flag = "False"

    pca_top_k = 10000


    triplet_embed_arg = dict(embed_batch_size = embed_batch_size, embed_dim = embed_dim, embed_epoch = embed_epoch,
                             hidden_size = hidden_size, layers = layers, dropout = dropout)
    embed_arg = dict(min_count=min_count_range[0], voc_size=voc_size_range[0], sg=sg_range[0],
                     negative=negative_range[0], sample=sample_range[0], hs=hs_range[0],
                     iter=iter_range[0], window=window_range[0], sentence_length=max(sentence_length_range))
    detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epochs_d_range,
                      lstm_unit_range=lstm_unit_range, optimizer_range=optimizer_range,
                      layer_range=layer_range, drop_out_range=drop_out_range,
                      learning_rate_range = learning_rate_range,gru_unit_range=gru_unit_range,
                      dense_unit_range=dense_unit_range, pool_size_range=pool_size_range,
                      kernel_size_range=kernel_size_range)

    merged_args = Merge(embed_arg, detect_arg)
    # os.system("pwd")
    # all_src_vec_part, all_src_label_part = process_all_data(pre_model_path, "src", embed_arg, cls, src_doc_path, rank_file, K_fold, mul_bin_flag, retrain)
    # all_ir_vec_part, all_ir_label_part = process_all_data(pre_model_path, "ir_id_1", embed_arg, cls, ir_doc_path, rank_file, K_fold, mul_bin_flag, retrain)
    # all_byte_vec_part, all_byte_label_part = process_all_data(pre_model_path, "byte_id_1", embed_arg, cls, byte_doc_path, rank_file, K_fold, mul_bin_flag, retrain)

    # all_src_vec_part, all_src_label_part, all_ir_vec_part, all_ir_label_part, all_byte_vec_part, all_byte_label_part = process_hybrid_data(pre_model_base, args.src_embed, args.ir_embed,
    #                                                                                                                                        args.byte_embed, embed_arg, cls, doc_base,
    #                                                                                                                                        K_fold, mul_bin_flag, args.retrain, args.src_code,
    #                                                                                                                                       args.ir_code, args.byte_code, args.conn)


    temp = "./"+args.embedding_network+"/"+args.cls + "/"
    if not os.path.isdir("./"+args.embedding_network):
        os.mkdir("./"+args.embedding_network)
    if not os.path.isdir(temp):
        os.mkdir(temp)
    save_base = temp + args.conn
    if not os.path.isdir(save_base):
        os.mkdir(save_base)
    save_base = save_base+"/"+str(triplet_embed_arg["embed_epoch"])+"_"+str(triplet_embed_arg["hidden_size"])+"_"+str(triplet_embed_arg["layers"])
    if not os.path.isdir(save_base):
        os.mkdir(save_base)
    # if args.hybrid == "pca":
    #     pca_test(all_src_vec_part, all_src_label_part, all_ir_vec_part, all_ir_label_part, all_byte_vec_part, all_byte_label_part,
    #              embed_arg, detect_arg, times, args.split_test, K_fold, pca_top_k, mul_bin_flag, args.neural, args.conn, save_base)
    # elif args.hybrid == "dir":
    #     direct(all_src_vec_part, all_src_label_part, all_ir_vec_part, all_byte_vec_part, embed_arg, detect_arg, times,
    #            args.split_test, K_fold, mul_bin_flag, args.neural, args.conn, save_base)
    # elif args.hybrid == "3loss":
        # triplet_loss_compare(all_src_vec_part, all_src_label_part, all_ir_vec_part, all_ir_label_part, all_byte_vec_part,
        #                      all_byte_label_part, class_num, times, K_fold, args.split_test, categorical_flag, embed_arg, detect_arg,
        #                      triplet_embed_arg, args.neural, mul_bin_flag, args.conn, save_base, args.embedding_network, args.src_embed)
    if args.hybrid == "3loss":
        triplet_loss_2(doc_base+"src", doc_base+"rank", class_num, times, K_fold, args.split_test, categorical_flag, embed_arg,
                       detect_arg, triplet_embed_arg, args.src_embed, args.neural,
                       mul_bin_flag, args.conn, save_base, args.embedding_network, pre_model_base)
