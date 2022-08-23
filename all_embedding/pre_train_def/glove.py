from gensim.models import word2vec
import sys
sys.path.append("../..")
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
from pre_train_def.utils import *

def gen_corpus(code_blocks, corpus_path):
    f = open(corpus_path, "wb+")
    f.truncate()
    f.close()
    test = []
    with open(corpus_path, "a") as f:
        for block in code_blocks:
            f.write(" ".join(block).replace("\n ", "").strip())
            test.append(" ".join(block))
            f.write("\n")
    return test

def gen_vocab(code_blocks, vocab_path):
    f = open(vocab_path, "wb+")
    f.truncate()
    f.close()
    sorted_tokens = {}
    test = []
    for block in code_blocks:
        for token in block:
            if token not in sorted_tokens.keys():
                sorted_tokens[token] = 1
            else:
                sorted_tokens[token] += 1
    sorted_tokens = sorted(sorted_tokens.items(), key=lambda x:x[1], reverse=False)
    with open(vocab_path, "a") as f:
        for value in sorted_tokens[::-1]:
            f.write(value[0] + " " + str(value[1]) + "\n")
            test.append(value[0] + " " + str(value[1]) + "\n")
    return test

def train_glove(args_range):
    corpus = args_range["corpus"]
    vocab_file = args_range["vocab_file"]
    cooccurrence_file = args_range["cooccurrence_file"]
    cooccurrence_shuf_file = args_range["cooccurrence_shuf_file"]
    build_dir = args_range["build_dir"]
    save_file = args_range["save_file"]
    memory = args_range["memory"]
    num_threads = args_range["num_threads"]
    verbose = args_range["verbose"]
    binary = args_range["binary"]
    glove_file = args_range["glove_file"]
    w2v_file = args_range["w2v_file"]

    os.system(build_dir + "/vocab_count -min-count " + str(args_range["vocab_min_count"]) + " -verbose " + str(
        verbose) + " < " + corpus + " > " + vocab_file)
    os.system(build_dir + "/cooccur -memory " + str(memory) + " -vocab-file " + vocab_file + " -verbose " + str(
        verbose) + " -window-size " + str(args_range["window_size"]) + " < " + corpus + " > " + cooccurrence_file)
    os.system(build_dir + "/shuffle -memory " + str(memory) + " -verbose " + str(
        verbose) + " < " + cooccurrence_file + " > " + cooccurrence_shuf_file)
    os.system(build_dir + "/glove -save-file " + save_file + "" + " -threads " + str(
        num_threads) + " -input-file " + cooccurrence_shuf_file \
              + " -x-max " + str(args_range["x_max"]) + " -iter " + str(args_range["max_iter"]) + " -vector-size " + str(
        args_range["vector_size"]) + " -binary " + str(binary) + " -vocab-file " \
              + vocab_file + " -verbose " + str(verbose))
    glove2word2vec(glove_file, w2v_file)

def train(dataset, code, base, pre_model_path, embed_arg, retrain):
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
        print("training src...")
        code_path = src_path
        corpus_path = base + "src_code/src_corpus.txt"
        vocab_path = base + "src_code/src_vocab.txt"
        cooccurrence_file = base + "src_code/cooccurrence.bin"
        cooccurrence_shuf_file = base + "src_code/cooccurrence.shuf.bin"
        build_dir = base + "build"
        save_file = base + "src_code/src_vectors"
        glove_file = base + 'src_code/src_vectors.txt'


    arg = dict(vocab_min_count=embed_arg["vocab_min_count"], vector_size=embed_arg["voc_size"],
               max_iter=embed_arg["max_iter"],
               window_size=embed_arg["window_size"], x_max=embed_arg["x_max"], corpus=corpus_path,
               vocab_file=vocab_path,
               cooccurrence_file=cooccurrence_file, cooccurrence_shuf_file=cooccurrence_shuf_file, build_dir=build_dir,
               save_file=save_file, memory=embed_arg["memory"], num_threads=embed_arg["num_threads"],
               verbose=embed_arg["verbose"], binary=embed_arg["binary"],
               glove_file=glove_file, w2v_file=pre_model_path)

    code_blocks, code_label = read_file(code_path)
    corpus = gen_corpus(code_blocks, corpus_path)
    vocab = gen_vocab(code_blocks, vocab_path)
    train_glove(arg)