from gensim.models import word2vec
import sys
sys.path.append("../..")
import re
import collections
import numpy as np
from pre_train_def.utils import *
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def train_word2vec(all_Blocks, args, path):
    model = word2vec.Word2Vec(all_Blocks, sg=args["sg"], min_count=args["min_count"], size=args["voc_size"],
                              negative=args["negative"], sample=args["sample"], hs=args["hs"], iter=args["iter"],
                              window=args["window"])
    # print(model[":="])
    # print(model["lookupswitch"])
    # vec1 = np.asarray(model[""])
    model.wv.save(path)
    print("done")

def train(dataset, code, pre_model_path, embed_arg, retrain):
    if retrain == "True":
        if os.path.isfile(pre_model_path):
            os.remove(pre_model_path)
    elif retrain == "False":
        if os.path.exists(pre_model_path):
            return
    if dataset == "sard_bin":
        src_path = "../../pickle_object/sard/embedding/doc_src_embedding"
        ir_path = "../../pickle_object/sard/embedding/doc_IR_embedding"
        byte_path = "../../pickle_object/sard/embedding/doc_byte_embedding"

    elif dataset == "spot_bin" or dataset == "spot_mul":
        src_path = "../../pickle_object/spotbugs/embedding/src"
        ir_path = "../../pickle_object/spotbugs/embedding/ir"
        byte_path = "../../pickle_object/spotbugs/embedding/byte"

    elif dataset == "oop_bin":
        src_path = "../../pickle_object/oopsla/embedding/src"
        ir_path = "../../pickle_object/oopsla/embedding/ir"
        byte_path = "../../pickle_object/oopsla/embedding/byte"
    else:
        print("training embedding parameter error")
        exit(0)

    if code == "src":
        print("train src...")
        doc_path, save_path, flag = src_path, pre_model_path, "src"
        code_blocks, code_label = read_file(doc_path)
        train_word2vec(code_blocks, embed_arg, save_path)
    elif "ir" in code:
        print("train ir...")
        doc_path, save_path,flag = ir_path, pre_model_path, "ir"
        code_blocks, code_label = read_ir_file(doc_path)
        train_word2vec(code_blocks, embed_arg, save_path)
    elif "byte" in code:
        print("train byte...")
        doc_path, save_path, flag = byte_path, pre_model_path, "byte"
        code_blocks, code_label = read_file(doc_path)
        train_word2vec(code_blocks, embed_arg, save_path)
    del code_blocks
    del code_label
    gc.collect()