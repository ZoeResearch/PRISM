import os
import tensorflow as tf
from appUtils import load_embed_model
from Util.utils import get_vec_concat
import collections
import pickle
import random
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def spotbug_predict(embed_base, embed_name, detect_name, times, all_code_docs, save_base):
    detect_model_path = "./score/" + embed_name + "_" + detect_name + "/best_model_" + str(times) + ".h5"
    # if not os.path.isdir(detect_model_path):
    #     detect_model_path = "./score/" + embed_name + "_" + detect_name + "/best_model_0.h5"
    best_detect_model = tf.keras.models.load_model(detect_model_path)
    embed_model = load_embed_model(embed_name, embed_base)
    if embed_name in ["w2v", "fasttext"]:
        voc_size, sentence_length = 100, 200
    elif embed_name == "glove":
        voc_size, sentence_length = 50, 200
    elif embed_name == "elmo":
        voc_size, sentence_length = 256, 200
    else:
        voc_size, sentence_length = 0, 0
        print("no embed_name!")
        exit(0)

    x_test = get_vec_concat(embed_model, all_code_docs, voc_size, sentence_length, operator_set=None, ignore_list=None, regulate_byte_flag="False")
    y_prob = best_detect_model.predict(x_test)
    # y_prob = get_average()
    print("predict finished")
    assert len(all_code_docs) == len(x_test) == len(y_prob)
    pred_data = list(zip(all_code_docs, y_prob))
    pred_data = sorted(pred_data, key=lambda k: k[1], reverse=True)
    pickle.dump(pred_data[:1000], save_base+"spotbugs_sort")
    return

if __name__ == '__main__':
    embed_base = "./model/src_top_20/"
    # embed_base = "../pre_train_def/spotbugs"
    save_base = "./data/sort/"
    data_base = "../pickle_object/spotbugs/embedding/src"
    all_code_docs = pickle.load(open(data_base, "rb"))
    embed_name = ["w2v"]
    detect_name = ["bgru"]
    if not os.path.exists(save_base):
        os.mkdir(save_base)
    # for k in range(1):

    random_list = random.sample(range(len(all_code_docs)), 10000)


    spotbug_predict(embed_base, embed_name[0], detect_name[0], 0, all_code_docs, save_base)
