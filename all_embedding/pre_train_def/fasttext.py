import sys
sys.path.append("../..")
from pre_train_def.utils import read_all_file
from gensim.models import fasttext
import os
import gc

def train_fasttext(all_IRBlocks, args, path):
    model = fasttext.FastText(sentences=all_IRBlocks, sg=args["sg"], min_count=args["min_count"], size=args["voc_size"],
                              negative=args["negative"], sample=args["sample"], hs=args["hs"], iter=args["iter"],
                              window=args["window"], workers=3)
    # except UnicodeEncodeError:
    #     print("error")
    model.wv.save(path)
    print(model[":="])
    print(model["CLS0"])
    print(model["CLS1"])
    # model.save(path)

def train(pre_model_path, embed_arg, retrain, doc_path):
    if retrain == "True":
        if os.path.isfile(pre_model_path):
            os.remove(pre_model_path)
    elif retrain == "False":
        if os.path.exists(pre_model_path):
            return

    code_blocks, code_label = read_all_file(doc_path)
    train_fasttext(code_blocks, embed_arg, pre_model_path)

    del code_blocks
    del code_label
    gc.collect()

