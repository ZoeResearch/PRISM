import sys
sys.path.append("../..")
from all_embedding.pre_train_def.utils import *
from gensim.models import fasttext
import os

def train_fasttext(all_IRBlocks, args, path):
    model = fasttext.FastText(sentences=all_IRBlocks, sg=args["sg"], min_count=args["min_count"], size=args["voc_size"],
                              negative=args["negative"], sample=args["sample"], hs=args["hs"], iter=args["iter"],
                              window=args["window"], workers=3)
    model.wv.save(path)
    print(model[":="])
    print(model["CLS0"])
    print(model["CLS1"])

def train(dataset, code, pre_model_path, embed_arg, retrain):
    if retrain == "True":
        if os.path.isfile(pre_model_path):
            os.remove(pre_model_path)
    elif retrain == "False":
        if os.path.exists(pre_model_path):
            return
    if dataset == "sard_bin":
        src_path = "../../pickle_object/sard/embedding/doc_src_embedding"
    elif dataset == "spot_bin" or dataset == "spot_mul":
        src_path = "../../pickle_object/spotbugs/embedding/src"
    elif dataset == "oop_bin":
        src_path = "../../pickle_object/oopsla/embedding/src"
    else:
        print("training embedding parameter error")
        exit(0)
    if code == "src":
        print("train src...")
        doc_path, save_path = src_path, pre_model_path
        code_blocks, code_label = read_file(doc_path)
        train_fasttext(code_blocks, embed_arg, save_path)
    del code_blocks
    del code_label
    gc.collect()

