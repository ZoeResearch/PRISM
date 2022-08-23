import os
import sys
sys.path.append("../../")
from Bert.utils import *
from Bert.Bert_model import cherry_pick
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess=tf.compat.v1.Session(config=config)

def run_train(times, all_input, all_label_part, embed_arg, detect_arg, split_test, K, mul_bin_flag, classification_model, code, save_base, bert_model):
    for num in range(1,times):
        start = time.time()
        print("**************************" + str(num) + " time training**************************")
        cherry_pick(all_input, all_label_part, embed_arg, detect_arg, num,
                    split_test, K, mul_bin_flag, classification_model, code, save_base, bert_model)
        end = time.time()
        print("total time:", end - start)

if __name__ == "__main__":
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

    spot_multiclass_doc_path = "../../pickle_object/spotbugs/detect_mul/src/"
    spot_multiclass_rank_file = "../../pickle_object/spotbugs/detect_mul/rank/"
    spot_binary_doc_path = "../../pickle_object/spotbugs/detect_bin_sample_15000/" + args.code + "/"
    spot_binary_rank_file = "../../pickle_object/spotbugs/detect_bin_sample_15000/rank/"
    spotbugs_pre_model_path = "../../pre_train_def/spotbugs/bert/"


    model_name_range = ["bert-base-uncased"]
    # model_name_range = ["microsoft/codebert-base"]
    max_length_range = [200]   #encode plus data have start and end
    learning_rate_range = [3e-5]
    # learning_rate_range = [3e-5, 5e-5, 8e-5]
    batch_size_range = [16]
    # batch_size_range = [32, 128]
    epoch_range = [40]
    # epoch_range = [50]
    optimizer_range = ["Adam"]

    if args.cls == "spot_bin":
        mul_bin_flag, doc_path, pre_model_path, rank_file = 0, spot_binary_doc_path, spotbugs_pre_model_path, spot_binary_rank_file
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

    detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range = epoch_range,
                      optimizer_range=optimizer_range, learning_rate_range=learning_rate_range)

    if args.bert_model == "bert_seq":
        for model_name in model_name_range:
            for max_length in max_length_range:
                embed_arg = dict(model_name = model_name, max_length = max_length)
                all_input_ids, all_token_type_ids, all_attention_mask, all_label_part = prepare_bert_data(doc_path, rank_file, int(args.fold), mul_bin_flag, model_name, max_length)
                run_train(times, [all_input_ids, all_token_type_ids, all_attention_mask], all_label_part, embed_arg, detect_arg, args.split_test, int(args.fold), mul_bin_flag,
                          args.neural, args.code, save_base, args.bert_model)
