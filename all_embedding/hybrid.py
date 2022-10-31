import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import gc
from pre_train_def import w2v
from Util.utils import *

torch.multiprocessing.set_sharing_strategy('file_system')
class MNIST(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.images = df.iloc[:, 1:].values.astype(np.uint8)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values
        else:
            self.images = df.values.astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item].reshape(28, 28, 1)

        if self.is_train:
            anchor_label = self.labels[item]

            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(28, 28, 1)

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(28, 28, 1)

            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
                positive_img = self.transform(self.to_pil(positive_img))
                negative_img = self.transform(self.to_pil(negative_img))

            return anchor_img, positive_img, negative_img, anchor_label

        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

def process_all_data(pre_model_path, code, embed_arg, cls, doc_path, rank_file, K_fold, mul_bin_flag, retrain):
    pre_model_path = pre_model_path + code + "_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size']) + ".wordvectors"
    vec_path = cls + "/" + "vec_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size'])
    label_path = cls + "/" + "label_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size'])
    # if os.path.exists(vec_path) and os.path.exists(label_path):
    #     print("loading...")
    #     all_vec_part, all_label_part = pickle.load(open(vec_path, "rb")), pickle.load(open(label_path, "rb"))
    # else:
    w2v.train(cls, code, pre_model_path, embed_arg, retrain)
    model = KeyedVectors.load(pre_model_path, mmap="r")
    all_vec_part, all_label_part = prepare_data(doc_path, model, embed_arg["voc_size"],
                                                embed_arg["sentence_length"], code, rank_file, K_fold,
                                                mul_bin_flag)
        # dump_object(vec_path, all_vec_part)
        # dump_object(label_path, all_label_part)
    print("load finished!")
    del model
    gc.collect()
    return all_vec_part, all_label_part

# fasttext glove bert
# rnn > mlp and cnn

if __name__ == '__main__':
    iter_range = [5]
    window_range = [5]
    sg_range = [0]
    min_count_range = [0]
    voc_size_range = [100]
    negative_range = [5]
    sample_range = [1e-3]
    hs_range = [0]
    sentence_length_range = [200]

    times = 0
    cls = "spot_bin"

    base = "./"
    pre_model_path = base+"pre_train_def/spotbugs/w2v/"
    src_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000/src/"
    ir_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000/ir_id_1/"
    byte_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000/byte_id_1/"
    rank_file = base+"pickle_object/spotbugs/detect_bin_sample_15000/rank/"

    K_fold = 11
    mul_bin_flag = 0
    class_num = 2
    retrain = "False"
    categorical_flag = "False"
    split_flag = "True"

    embed_arg = dict(min_count=min_count_range[0], voc_size=voc_size_range[0], sg=sg_range[0],
                     negative=negative_range[0], sample=sample_range[0], hs=hs_range[0],
                     iter=iter_range[0], window=window_range[0], sentence_length=sentence_length_range[0])
    all_src_vec_part, all_src_label_part = process_all_data(pre_model_path, "src", embed_arg, cls, src_doc_path, rank_file, K_fold, mul_bin_flag, retrain)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(all_src_vec_part, all_src_label_part, class_num,
                                                                   times, K_fold, split_flag, categorical_flag)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(all_src_vec_part, all_src_label_part, class_num,
                                                                   times, K_fold, split_flag, categorical_flag)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(all_src_vec_part, all_src_label_part, class_num,
                                                                   times, K_fold, split_flag, categorical_flag)



