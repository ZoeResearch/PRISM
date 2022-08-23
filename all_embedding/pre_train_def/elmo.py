import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import scipy.spatial.distance as ds
import pickle
import sys
sys.path.append("..")
from bilm_tf.bilm import Batcher, BidirectionalLanguageModel, weight_layers
from bilm_tf.bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm_tf.bilm.data import BidirectionalLMDataset
from bilm_tf.bilm.training import dump_weights
import collections
import shutil
import json
import copy
import time
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def process_corpus(all_code, dir):
    for i in range(0, len(all_code), 10):
        text = "\n".join(all_code[i:i + 10])
        fp = open(dir + "/" + str(i) + ".txt", "w")
        fp.write(text)
        fp.close()

def Counter(all_code, dir):
    print("dir:", dir)
    all_code = " ".join(all_code).split()
    dic = {}
    for i in all_code:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    sort_dic = sorted(dic.items(), key=lambda kv: (kv[1], kv[0]))
    sort_words = []
    for i in reversed(sort_dic):
        # if i[1] != 1:
        sort_words.append(i[0])  # 添加if的话会排除了出现一次的单词 min_count=1

    sorted_vocab = ["<S>", "</S>", "<UNK>"]
    # sorted_vocab.extend([pair[0] for pair in dictionary.most_common()])
    sorted_vocab.extend(sort_words)
    with open(dir, "w") as f:
        f.write("\n".join(sorted_vocab))

    return len(sorted_vocab)

def load_code(code_path):
    embed_doc = pickle.load(open(code_path, "rb"))
    embed_code = []
    for doc in embed_doc:
        embed_code.append(doc.words)
    token_num = 0
    for i in range(len(embed_code)):
        embed_code[i] = " ".join(embed_code[i]) + " ."
        token_num += len(embed_code[i].split())
    print("token number:", token_num)
    return embed_code, token_num

def dump_embedding(vocab_file, model_path, code_path):
    with tf.device('/cpu:0'):
        # Location of pretrained LM.  Here we use the test fixtures.
        options_file = os.path.join(model_path, 'options2.json')
        weight_file = os.path.join(model_path, 'weight.hdf5')
        result_embed = os.path.join(model_path, 'model.pickle')
        if os.path.exists(result_embed):
            os.remove(result_embed)

        # Create a Batcher to map text to character ids. 长度为50
        batcher = Batcher(vocab_file, 50)

        # Input placeholders to the biLM.
        context_character_ids = tf.placeholder('int32', shape=(None, None, 50))

        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(options_file, weight_file)

        # Get ops to compute the LM embeddings.
        context_embeddings_op = bilm(context_character_ids)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

        word_embedding = {}
        embed_doc = pickle.load(open(code_path, "rb"))
        tokenized_context = []

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        for i in range(len(embed_doc)):
            tokenized_context.append(embed_doc[i].words)
            if i != 0 and i % 10 == 0:   #100?????
                # Create batches of data.
                context_ids = batcher.batch_sentences(tokenized_context)
                print("Shape of context ids = ", context_ids.shape)

                # Compute ELMo representations (here for the input only, for simplicity).
                elmo_context_input_ = sess.run(
                    elmo_context_input['weighted_op'],
                    feed_dict={context_character_ids: context_ids}
                )
                print("Shape of generated embeddings = ", elmo_context_input_.shape)
                for k in range(len(tokenized_context)):  # every sentence
                    for j in range(len(tokenized_context[k])):
                        if tokenized_context[k][j] not in word_embedding.keys():
                            word_embedding[tokenized_context[k][j]] = elmo_context_input_[k][j]
                tokenized_context = []
    with open(result_embed, "wb") as f:
        pickle.dump(word_embedding, f)

    test = pickle.load(open(result_embed, "rb"))
    print(test["("])

def train_elmo(n_train_tokens, vocab, data, src_model_path):
    n_gpus = 1
    vector_dim = 16
    filter = [[1, 32],
              [2, 32],
              [3, 64],
              [4, 128],
              [5, 256],
              [6, 512],
              [7, 1024]]
    max_characters_per_token = 50
    n_characters = 261
    n_highway = 1
    dropout = 0.1
    lstm = {
        'cell_clip': 3,
        'dim': 1024,
        'n_layers': 2,
        'proj_clip': 3,
        'projection_dim': 128,
        'use_skip_connections': True}
    all_clip_norm_val = 10.0
    n_epochs = 10
    batch_size = 128
    n_tokens_vocab = vocab.size
    unroll_steps = 20
    n_negative_samples_batch = 8192

    options = {
        'bidirectional': True,
        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': vector_dim},
                     'filters': filter,
                     'max_characters_per_token': max_characters_per_token,
                     'n_characters': n_characters,
                     'n_highway': n_highway},
        'dropout': dropout,
        'lstm': lstm,
        'all_clip_norm_val': all_clip_norm_val,
        'n_epochs': n_epochs,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': unroll_steps,
        'n_negative_samples_batch': n_negative_samples_batch,

    }
    options2 = copy.deepcopy(options)
    options2['char_cnn']['n_characters'] = 262
    with open(os.path.join(src_model_path, 'options2.json'), 'w') as fout:
        fout.write(json.dumps(options2))

    train(options, data, n_gpus, src_model_path, src_model_path)


if __name__ == '__main__':
    data = "spot"

    if data == "sard":
        src_path = "../pickle_object/sard/embedding/doc_src_embedding"
        ir_path = "../pickle_object/sard/embedding/doc_IR_embedding"
        byte_path = "../pickle_object/sard/embedding/doc_byte_embedding"
        src_model_path = "./sard/elmo/src_ckpt/"
        ir_model_path = "./sard/elmo/ir_ckpt/"
        byte_model_path = "./sard/elmo/byte_ckpt/"
        src_corpus, ir_corpus, byte_corpus = "./sard/elmo/src_train_corpus", "./sard/elmo/ir_train_corpus", "./sard/elmo/byte_train_corpus"
        src_vocab, ir_vocab, byte_vocab = "./sard/elmo/src_vocab.txt", "./sard/elmo/ir_vocab.txt", "./sard/elmo/byte_vocab.txt"
    elif data == "spot":
        src_path = "../pickle_object/spotbugs/embedding/src"
        ir_path = "../pickle_object/spotbugs/embedding/ir"
        byte_path = "../pickle_object/spotbugs/embedding/byte"
        src_model_path = "./spotbugs/elmo/src_ckpt/"
        ir_model_path = "./spotbugs/elmo/ir_ckpt/"
        byte_model_path = "./spotbugs/elmo/byte_ckpt/"
        src_corpus, ir_corpus, byte_corpus = "./spotbugs/elmo/src_train_corpus", "./spotbugs/elmo/ir_train_corpus", "./spotbugs/elmo/byte_train_corpus"
        src_vocab, ir_vocab, byte_vocab = "./spotbugs/elmo/src_vocab.txt", "./spotbugs/elmo/ir_vocab.txt", "./spotbugs/elmo/byte_vocab.txt"
    elif data == "oop":
        src_path = "../pickle_object/oopsla/embedding/src"
        ir_path = "../pickle_object/oopsla/embedding/ir"
        byte_path = "../pickle_object/oopsla/embedding/byte"
        src_model_path = "./oopsla/elmo/src_ckpt/"
        ir_model_path = "./oopsla/elmo/ir_ckpt/"
        byte_model_path = "./oopsla/elmo/byte_ckpt/"
        src_corpus, ir_corpus, byte_corpus = "./oopsla/elmo/src_train_corpus", "./oopsla/elmo/ir_train_corpus", "./oopsla/elmo/byte_train_corpus"
        src_vocab, ir_vocab, byte_vocab = "./oopsla/elmo/src_vocab.txt", "./oopsla/elmo/ir_vocab.txt", "./oopsla/elmo/byte_vocab.txt"

    for path in [src_model_path, ir_model_path, byte_model_path, src_corpus, ir_corpus, byte_corpus]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    for path in [src_vocab, ir_vocab, byte_vocab]:
        if os.path.exists(path):
            os.remove(path)

    print("processing src...")
    # preprocess, generate vocab and data
    embed_code, token_num = load_code(src_path)
    process_corpus(embed_code, src_corpus)
    vocab_size = Counter(embed_code, src_vocab)

    # load data and vocab
    vocab = load_vocab(src_vocab, 50)
    data = BidirectionalLMDataset(src_corpus + "/*", vocab, test=False,
                                  shuffle_on_load=True)
    # training with different parameters
    print("training...")
    train_elmo(token_num, vocab, data, src_model_path)
    # save all embeddings
    print("dump weight...")

    dump_weights(src_model_path, src_model_path + "weight.hdf5")
    #
    start = time.time()
    print("dump embedding...")
    dump_embedding(src_vocab, src_model_path, src_path)
    end = time.time()
    print("total:", end-start)


















