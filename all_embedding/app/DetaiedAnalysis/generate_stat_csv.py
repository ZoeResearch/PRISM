import collections
import os
import pickle
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def generate_csv(part_sort_pattern, save_path, number):
    pattern_name = [pattern[0] for pattern in part_sort_pattern[-1]]
    for i in range(len(part_sort_pattern)):
        sort_pattern = part_sort_pattern[i]
        dic = {}
        new_dic = collections.OrderedDict()
        for item in sort_pattern:
            dic[item[0]] = item[1]
        for name in pattern_name:
            if name not in dic.keys():
                new_dic[name] = 0
            else:
                new_dic[name] = dic[name] / number[i]
        write_csv(new_dic, save_path +"/draw.csv")

def write_csv(score, file_name):
    if os.path.exists(file_name):
        f = open(file_name, "a")
    else:
        f = open(file_name, "a")
        for i in score.keys():
            f.write(str(i))
            f.write("\t")
        f.write("\n")
    for i in score.values():
        f.write(str(i))
        f.write("\t")
    f.write("\n")
    f.close()


if __name__ == '__main__':
    number = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000, 6000, 8000, 10000, 1572]
    model_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm',
                  'fasttext_blstm', 'glove_bgru', 'fasttext_textcnn']
    for single_name in model_name:
        save_base = "../score/" + single_name + "/"
        path = save_base + "/data_0/"
        model = save_base + "best_model_0.h5"
        if os.path.exists(path+"sort_pattern.pkl"):
            train, val, test, pattern, ranked_doc = pickle.load(open(path+"train_doc.pkl", "rb")), \
                                        pickle.load(open(path+"val_doc.pkl", "rb")), \
                                        pickle.load(open(path+"test_doc.pkl", "rb")), \
                                        pickle.load(open(path+"sort_pattern.pkl", "rb")),\
                                        pickle.load(open(path+"ranked_test_doc.pkl", "rb"))
            #
            # generate_csv(pattern, path, number)
            # print(path)


            print("done!")

