import pickle
import os
import collections
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')


if __name__ == '__main__':
    doc_path = "/data/bugDetection/srcIRcom/all_embedding/pickle_object/spotbugs/embedding/src"
    all_code = pickle.load(open(doc_path, "rb"))
    print(len(all_code))




























