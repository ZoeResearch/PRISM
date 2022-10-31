
from pre_train_def import w2v, fasttext, glove
from Util.utils import *
from Util.gen_BYTE_vec import allIns
from tensorflow.python.keras.utils import np_utils

def process2vec(pre_model_path, code, embed_model, embed_arg, cls, doc_path, rank_file, K_fold, mul_bin_flag, retrain):
    # if code == "ir_id_1" or code == "byte_id_1":
    #     temp = code.split("_")[0]
    # else:
    #     temp = code
    temp = code
    pre_model_path = pre_model_path + "/" + temp + "_" + str(embed_arg['iter']) + "_" + str(
            embed_arg['window']) + "_" + str(embed_arg['voc_size']) + ".wordvectors"
    print(pre_model_path)
    vec_path = "Word2vec/src_code/" + cls + "/" + temp + "_vec_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size'])
    label_path = "Word2vec/src_code/" + cls + "/" + temp + "_label_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size'])
    # if os.path.exists(vec_path) and os.path.exists(label_path):
    #     print("loading...")
    #     all_vec_part, all_label_part = pickle.load(open(vec_path, "rb")), pickle.load(open(label_path, "rb"))
    # else:
    # w2v.train(cls, code, pre_model_path, embed_arg, retrain)
    if embed_model == "w2v":
        w2v.train(cls, code, pre_model_path, embed_arg, retrain, allIns)
        model = KeyedVectors.load(pre_model_path, mmap="r")
    elif embed_model == "fasttext":
        fasttext.train(cls, code, pre_model_path, embed_arg, retrain)
        model = KeyedVectors.load(pre_model_path, mmap="r")
    elif embed_model == "glove":
        embed_base = "./pre_train_def/GloVe-master/"
        glove.train(cls, code, embed_base, pre_model_path, embed_arg, retrain)
        model = KeyedVectors.load_word2vec_format(pre_model_path)
    elif embed_model == "elmo":
        model = pickle.load(open(pre_model_path, "rb"))
    all_vec_part, all_label_part = prepare_data(doc_path, model, embed_arg["voc_size"],
                                                embed_arg["sentence_length"], code, rank_file, K_fold,
                                                mul_bin_flag)
    # dump_object(vec_path, all_vec_part)
    # dump_object(label_path, all_label_part)
    print("load finished!")
    del model
    gc.collect()
    return all_vec_part, all_label_part

def process_hybrid_data(pre_model_base, src_embed, ir_embed, byte_embed, embed_arg, cls, doc_base, K_fold, mul_bin_flag, retrain, src, ir, byte, conn):
    src_model_base, ir_model_base, byte_model_base = pre_model_base+src_embed, pre_model_base+ir_embed, pre_model_base+byte_embed
    all_src_vec_part, all_src_label_part = process2vec(src_model_base, src, src_embed, embed_arg, cls, doc_base+src, doc_base+"rank", K_fold, mul_bin_flag, retrain)
    if not conn:
        print("no conn")
        exit(0)
    elif conn == "3loss_src":
        all_ir_vec_part, all_ir_label_part = [], []
        all_byte_vec_part, all_byte_label_part = [], []
    elif conn == "3loss_ir":
        all_ir_vec_part, all_ir_label_part = process2vec(ir_model_base, ir, ir_embed, embed_arg, cls, doc_base + ir,
                                                         doc_base + "rank", K_fold, mul_bin_flag, retrain)
        all_byte_vec_part, all_byte_label_part = [], []
    elif conn == "3loss_byte1" or conn == "3loss_byte2" or conn == "3loss_byte3":
        all_ir_vec_part, all_ir_label_part = [], []
        all_byte_vec_part, all_byte_label_part = process2vec(byte_model_base, byte, byte_embed, embed_arg, cls,
                                                             doc_base + byte, doc_base + "rank", K_fold, mul_bin_flag,
                                                             retrain)
    elif conn == "3loss_src_ir":
        all_ir_vec_part, all_ir_label_part = process2vec(ir_model_base, ir, ir_embed, embed_arg, cls, doc_base+ir, doc_base+"rank", K_fold, mul_bin_flag, retrain)
        all_byte_vec_part, all_byte_label_part = [], []
    elif conn == "3loss_src_byte1" or conn == "3loss_src_byte2" or conn == "3loss_src_byte3":
        all_ir_vec_part, all_ir_label_part = [], []
        all_byte_vec_part, all_byte_label_part = process2vec(byte_model_base, byte, byte_embed, embed_arg, cls, doc_base+byte, doc_base+"rank", K_fold, mul_bin_flag, retrain)
    else:
        all_ir_vec_part, all_ir_label_part = process2vec(ir_model_base, ir, ir_embed, embed_arg, cls, doc_base + ir,
                                                         doc_base + "rank", K_fold, mul_bin_flag, retrain)
        all_byte_vec_part, all_byte_label_part = process2vec(byte_model_base, byte, byte_embed, embed_arg, cls,
                                                             doc_base + byte, doc_base + "rank", K_fold, mul_bin_flag,
                                                             retrain)

    return all_src_vec_part, all_src_label_part, all_ir_vec_part, all_ir_label_part, all_byte_vec_part, all_byte_label_part
