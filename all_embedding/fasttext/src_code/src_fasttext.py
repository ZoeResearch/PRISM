import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
import sys
sys.path.append("../../")
from pre_train_def import fasttext
from Util.training import cherry_pick
from Util.utils import *
import time
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=config)


def process_all_data(pre_model_path, code, embed_arg, cls, doc_path, rank_file, K_fold, mul_bin_flag, retrain):
    pre_model_path = pre_model_path + args.code + "_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size']) + ".wordvectors"
    fasttext.train(pre_model_path, embed_arg, args.retrain, doc_path)
    model = KeyedVectors.load(pre_model_path, mmap="r")
    all_vec_part, all_label_part = prepare_data(doc_path, model, embed_arg["voc_size"],
                                                embed_arg["sentence_length"], code, rank_file, K_fold,
                                                mul_bin_flag)
    del model
    gc.collect()
    print("load finished!")
    return all_vec_part, all_label_part

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '-c', help='data category')
    parser.add_argument('--code', '-d', help='code category')
    parser.add_argument('--fold', '-k', help='the number of data of every fold')
    parser.add_argument("--retrain", "-r")
    parser.add_argument('--split_test', '-s')
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--neural', '-n')
    args = parser.parse_args()

    K_fold = int(args.fold)
    sard_binary_doc_path = "../../pickle_object/sard/detect/src/"
    sard_binary_rank_file = "../../pickle_object/sard/detect/rank/"
    spot_multiclass_doc_path = "../../pickle_object/spotbugs/detect_mul/" + args.code + "/"
    spot_multiclass_rank_file = "../../pickle_object/spotbugs/detect_mul/rank/"
    spot_binary_doc_path = "../../pickle_object/spotbugs/detect_bin_sample_15000/" + args.code + "/"
    spot_binary_rank_file = "../../pickle_object/spotbugs/detect_bin_sample_15000/rank/"
    oop_binary_doc_path, oop_pre_model_path, oop_binary_rank_file = "../../pickle_object/oopsla/detect_bin/" + args.code + "/", \
                                                                    "../../pre_train_def/oopsla/fasttext/", \
                                                                    "../../pickle_object/oopsla/detect_bin/rank/"

    sard_pre_model_path = "../../pre_train_def/sard/fasttext/"
    spotbugs_pre_model_path = "../../pre_train_def/spotbugs/fasttext/"

    iter_range = [5]
    window_range = [5]
    sg_range = [0]
    min_count_range = [0]
    voc_size_range = [100]
    negative_range = [5]
    sample_range = [1e-3]
    hs_range = [0]
    sentence_length_range = [200]

    batch_size_range = [32, 128]
    epochs_d_range = [40]
    lstm_unit_range = [16,32,64]     #fixed
    optimizer_range = ["Adam"] #fixed
    layer_range = [2,4]    #fixed
    drop_out_range = [0.2]
    learning_rate_range = [0.0005,0.001,0.002]

    gru_unit_range = [128, 256, 512]
    dense_unit_range = [32, 64, 128]
    pool_size_range = [5, 10, 20]
    kernel_size_range = [5, 10, 20]

    if args.cls == "sard_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, sard_binary_doc_path, sard_pre_model_path, sard_binary_rank_file
    elif args.cls == "spot_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, spot_binary_doc_path, spotbugs_pre_model_path, spot_binary_rank_file
    elif args.cls == "spot_mul":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 1, spot_multiclass_doc_path, spotbugs_pre_model_path, spot_multiclass_rank_file
    elif args.cls == "oop_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, oop_binary_doc_path, oop_pre_model_path, oop_binary_rank_file
    else:
        print("no category!")

    detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epochs_d_range,
                      lstm_unit_range=lstm_unit_range, optimizer_range=optimizer_range,
                      layer_range=layer_range, drop_out_range=drop_out_range,
                      learning_rate_range=learning_rate_range, gru_unit_range=gru_unit_range,
                      dense_unit_range=dense_unit_range, pool_size_range=pool_size_range,
                      kernel_size_range=kernel_size_range)
    print("tf.test.is_gpu_available():", tf.test.is_gpu_available())
    if not os.path.isdir(args.cls):
        os.mkdir(args.cls)
    save_base = args.cls + "/" + args.code + "_fasttext_" + args.neural
    if not os.path.isdir(save_base):
        os.mkdir(save_base)
    for iter in iter_range:
        for window in window_range:
            for voc in voc_size_range:
                print("preparing for data...")
                embed_arg = dict(min_count=min_count_range[0], voc_size=voc, sg=sg_range[0],
                                 negative=negative_range[0], sample=sample_range[0], hs=hs_range[0],
                                 iter=iter, window=window, sentence_length=sentence_length_range[0])
                all_vec_part, all_label_part = process_all_data(pre_model_path, args.code, embed_arg, args.cls,doc_path, rank_file, K_fold, mul_bin_flag, args.retrain)
                if args.split_test == "True":
                    num = K_fold - 1
                elif args.split_test == "False":
                    num = K_fold

                for times in range(num):
                    start = time.time()
                    print("**************************" + str(times) + " time training**************************")
                    cherry_pick(all_vec_part, all_label_part, embed_arg, detect_arg, times, args.split_test,
                                                       K_fold, mul_bin_flag, args.neural, args.code, save_base)
                    end = time.time()
                    print("total time:", end - start)
                del all_vec_part
                del all_label_part
                gc.collect()

