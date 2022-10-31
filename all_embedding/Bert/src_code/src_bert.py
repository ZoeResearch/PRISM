import os
import pickle

import sys
sys.path.append("../../")
from Bert.utils import *
from Bert.Bert_model import cherry_pick
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
config=tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=config)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_train(times, all_input, all_label_part, embed_arg, detect_arg, split_test, K, mul_bin_flag, classification_model, code, save_base, bert_model):
    for num in range(4, times):
    # for num in range(times):
        start = time.time()
        print("**************************" + str(num) + " time training**************************")
        cherry_pick(all_input, all_label_part, embed_arg, detect_arg, num,
                    split_test, K, mul_bin_flag, classification_model, code, save_base, bert_model)
        end = time.time()
        print("time:", end - start)

if __name__ == "__main__":
    if not (tf.test.is_gpu_available() and tf.test.is_built_with_cuda()):
        print("CUDA or GPU not available!")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '-c', help='data category')
    parser.add_argument('--code', '-d', help='code category')
    parser.add_argument('--fold', '-k', help='the number of data of every fold')
    parser.add_argument("--retrain", "-r")
    parser.add_argument('--split_test', '-s')
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--bert_model', '-p')
    parser.add_argument('--neural', '-n')

    args = parser.parse_args()

    sard_binary_doc_path = "../../pickle_object/sard/detect/src/"
    sard_binary_rank_file = "../../pickle_object/sard/detect/rank/"
    spot_multiclass_doc_path = "../../pickle_object/spotbugs/detect_mul/src/"
    spot_multiclass_rank_file = "../../pickle_object/spotbugs/detect_mul/rank/"
    spot_binary_doc_path = "../../pickle_object/spotbugs/detect_bin_sample_15000/" + args.code + "/"
    spot_binary_rank_file = "../../pickle_object/spotbugs/detect_bin_sample_15000/rank/"
    # spot_binary_doc_path = "../../pickle_object/spotbugs/detect_bin_sample_15000_test/" + args.code + "/"
    # spot_binary_rank_file = "../../pickle_object/spotbugs/detect_bin_sample_15000_test/rank/"
    # spot_binary_doc_path = "../../pickle_object/spotbugs/test/" + args.code + "/"
    # spot_binary_rank_file = "../../pickle_object/spotbugs/test/rank/"
    sard_pre_model_path = "../../pre_train_def/sard/bert/"
    spotbugs_pre_model_path = "../../pre_train_def/spotbugs/bert/"

    oop_binary_doc_path, oop_pre_model_path, oop_binary_rank_file = "../../pickle_object/oopsla/detect_bin/" + args.code + "/", \
                                                                    "../../pre_train_def/oopsla/w2v/", \
                                                                    "../../pickle_object/oopsla/detect_bin/rank/"
    # for spot_bin
    if args.cls == "spot_bin":
        max_length_range = [200]   #encode plus data have start and end
        # learning_rate_range = [1e-5, 1e-3, 0.1]
        learning_rate_range = [0.001]
        batch_size_range = [4]
        epoch_range = [80]
        encode_layer_range = [0,2]
        optimizer_range = ["Adam"]


        epochs_d_range = [50]
        # lstm_unit_range = [16, 32, 64]
        lstm_unit_range = [16]
        # layer_range = [2, 4, 6]
        layer_range = [6]
        drop_out_range = [0.2]
        # gru_unit_range = [64, 128, 256]
        gru_unit_range = [64]
        # dense_unit_range = [64, 128, 256]
        dense_unit_range = [64]
        # pool_size_range = [5, 10, 20]
        # pool_size_range = [10, 20]
        pool_size_range = [10]
        # kernel_size_range = [5, 10, 15]
        kernel_size_range = [5]
    elif args.cls == "spot_mul":
        # for spot-mul
        max_length_range = [200]  # encode plus data have start and end
        # learning_rate_range = [1e-3, 3e-5, 3e-1]
        learning_rate_range = [1e-3]
        batch_size_range = [4]
        epoch_range = [80]
        encode_layer_range = [0, 2]
        optimizer_range = ["Adam"]

        epochs_d_range = [50]
        # lstm_unit_range = [16, 32, 64]
        lstm_unit_range = [64]
        # layer_range = [2, 4, 6]
        layer_range = [2,4]
        # drop_out_range = [0.2, 0.5]
        drop_out_range = [0.2]
        # gru_unit_range = [16, 32, 128]
        gru_unit_range = [64]
        # dense_unit_range = [16, 64, 256]
        dense_unit_range = [16]
        # pool_size_range = [10, 15, 20]
        pool_size_range = [10]
        # kernel_size_range = [5, 10]
        kernel_size_range = [5]
    else:
        print("do not specify cls")
        exit(0)

    # epochs_d_range = [40]
    # lstm_unit_range = [16]  # fixed
    # # optimizer_range = ["Adam"]  # fixed
    # layer_range = [2]  # fixed
    # drop_out_range = [0.2]
    # # learning_rate_range = [0.0005, 0.001, 0.002]
    # gru_unit_range = [128]
    # dense_unit_range = [32]
    # pool_size_range = [5]
    # kernel_size_range = [5]

    if args.cls == "sard_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, sard_binary_doc_path, sard_pre_model_path, sard_binary_rank_file
    elif args.cls == "spot_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, spot_binary_doc_path, spotbugs_pre_model_path, spot_binary_rank_file
    elif args.cls == "oop_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, oop_binary_doc_path, oop_pre_model_path, oop_binary_rank_file
    elif args.cls == "spot_mul":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 1, spot_multiclass_doc_path, spotbugs_pre_model_path, spot_multiclass_rank_file
    else:
        print("no category!")
        exit(0)
    if not os.path.isdir(args.cls):
        os.mkdir(args.cls)
    save_base = args.cls + "/" + args.code + "_" + args.bert_model + "_" + args.neural
    if not os.path.isdir(save_base):
        os.mkdir(save_base)
    if args.split_test == "True":
        times = int(args.fold) - 1
    elif args.split_test == "False":
        times = int(args.fold)

    if args.bert_model == "bert_seq":
        detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epoch_range,
                          optimizer_range=optimizer_range, learning_rate_range=learning_rate_range,
                          encode_layer_range = encode_layer_range)
        model_name_range = ["bert-base-uncased"]
        for model_name in model_name_range:
            for max_length in max_length_range:
                embed_arg = dict(model_name = model_name, max_length = max_length)
                all_input_ids, all_token_type_ids, all_attention_mask, all_label_part = prepare_bert_model_data(doc_path, rank_file, int(args.fold), mul_bin_flag, model_name, max_length)
                run_train(times, [all_input_ids, all_token_type_ids, all_attention_mask], all_label_part, embed_arg, detect_arg, args.split_test, int(args.fold), mul_bin_flag,
                          args.neural, args.code, save_base, args.bert_model)
    # elif args.bert_model == "bert_token":

    elif args.bert_model == "codebert_seq":
        detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epoch_range,
                          optimizer_range=optimizer_range, learning_rate_range=learning_rate_range,
                          encode_layer_range = encode_layer_range)
        model_name_range = ["microsoft/codebert-base"]
        # tokenizer = RobertaTokenizer.from_pretrained("")
        for model_name in model_name_range:
            for max_length in max_length_range:
                embed_arg = dict(model_name = model_name, max_length = max_length)
                all_input_ids, all_token_type_ids, all_attention_mask, all_label_part = prepare_roberta_model_data(doc_path, rank_file, int(args.fold), mul_bin_flag, model_name, max_length, model_name)
                for num in range(times):
                    start = time.time()
                    print("**************************" + str(num) + " time training**************************")
                    cherry_pick([all_input_ids, all_token_type_ids, all_attention_mask], all_label_part, embed_arg, detect_arg, num,
                                args.split_test, int(args.fold), mul_bin_flag, args.neural, args.code, save_base, args.bert_model, model_name)
                    end = time.time()
                    print("total time:", end - start)
    elif args.bert_model == "roberta_seq":
        detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epoch_range,
                          optimizer_range=optimizer_range, learning_rate_range=learning_rate_range,
                          encode_layer_range = encode_layer_range)
        model_name_range = ["roberta-base"]
        # tokenizer = RobertaTokenizer.from_pretrained("")
        for model_name in model_name_range:
            for max_length in max_length_range:
                embed_arg = dict(model_name=model_name, max_length=max_length)
                all_input_ids, all_token_type_ids, all_attention_mask, all_label_part = prepare_roberta_model_data(
                    doc_path, rank_file, int(args.fold), mul_bin_flag, model_name, max_length, model_name)
                for num in range(times):
                # for num in range(1):
                    start = time.time()
                    print("**************************" + str(num) + " time training**************************")
                    cherry_pick([all_input_ids, all_token_type_ids, all_attention_mask], all_label_part, embed_arg,
                                detect_arg, num,
                                args.split_test, int(args.fold), mul_bin_flag, args.neural, args.code, save_base,
                                args.bert_model, model_name)
                    end = time.time()
                    print("total time:", end - start)

    elif args.bert_model == "codebert_token" or args.bert_model == "roberta_token" or args.bert_model == "bert_token":
        detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epochs_d_range,
                          lstm_unit_range=lstm_unit_range, optimizer_range=optimizer_range,
                          layer_range=layer_range, drop_out_range=drop_out_range,
                          learning_rate_range=learning_rate_range, gru_unit_range=gru_unit_range,
                          dense_unit_range=dense_unit_range, pool_size_range=pool_size_range,
                          kernel_size_range=kernel_size_range)
        if args.bert_model == "codebert_token":
            model_name = "microsoft/codebert-base"
            data_name = args.cls + "/input_of_codebert_token.pkl"
        elif args.bert_model == "roberta_token":
            model_name = "roberta-base"
            data_name = args.cls + "/input_of_roberta_token.pkl"
        elif args.bert_model == "bert_token":
            model_name = "bert-base-uncased"
            data_name = args.cls + "/input_of_bert_token.pkl"
        else:
            print("not specify bert model!")
            exit(0)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embed_arg = dict(model_name=model_name, sentence_length=200, voc_size=768)

        if os.path.exists(data_name):
            all_vec_part, all_label_part = pickle.load(open(data_name, "rb"))
        else:
            all_vec_part, all_label_part = get_codebert_word_input(doc_path, rank_file, int(args.fold), mul_bin_flag, tokenizer, model, embed_arg["sentence_length"], embed_arg["voc_size"])
            pickle.dump([all_vec_part, all_label_part], open(data_name, "wb"))
        del tokenizer
        gc.collect()
        run_train(10, all_vec_part, all_label_part, embed_arg, detect_arg, args.split_test, int(args.fold), mul_bin_flag,
                          args.neural, args.code, save_base, args.bert_model)

