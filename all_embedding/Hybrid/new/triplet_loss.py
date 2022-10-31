import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from hybrid_utils import *
from Util.utils import prepare_data, get_score_binaryclassfication, get_score_multiclassfication, split_dataset, split_codedoc, get_label
from Util.training import cherry_pick_new
from app.appUtils import load_embed_model
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

class CodeData(Dataset):
    def __init__(self, src_vec, ir_vec, byte_vec, labels, conn, train=True):
        self.is_train = train
        self.src_vec = src_vec
        # self.ir_vec = ir_vec
        # self.byte_vec = byte_vec
        self.labels = labels
        # for i in src_vec.keys():
        #     self.src_vec += src_vec[i]
        #     self.ir_vec += ir_vec[i]
        #     self.byte_vec += byte_vec[i]
        # assert len(src_vec)==len(ir_vec)==len(byte_vec)
        self.index = np.asarray([i for i in range(len(self.src_vec))])
        if conn == "3loss_src_ir":
            self.aux_vec = ir_vec
        elif conn == "3loss_src_byte1" or conn == "3loss_src_byte2" or conn == "3loss_src_byte3":
            self.aux_vec = byte_vec

    def __len__(self):
        return len(self.src_vec)

    # def __getitem__(self, item):
    #     # anchor--src pos--ir neg--other label src
    #     # anchor_code = self.src_vec[item]
    #     anchor_code = self.aux_vec[item]
    #     anchor_label = self.labels[item]
    #     # positive_code = self.aux_vec[item]
    #     positive_code = self.src_vec[item]
    #     # negative_list = [i for i in range(len(self.src_vec)) if self.labels[i] != anchor_label]
    #     negative_list = [i for i in range(len(self.src_vec)) if self.labels[i] != anchor_label]
    #     negative_item = random.choice(negative_list)
    #     negative_code = self.src_vec[negative_item]
    #
    #     return anchor_code, positive_code, negative_code, anchor_label

    def __getitem__(self, item):
        # anchor--src pos--ir neg--other label src
        anchor_code = self.aux_vec[item]
        anchor_label = self.labels[item]
        positive_code = self.src_vec[item]
        negative_list = [i for i in range(len(self.src_vec)) if i!=item]
        negative_item = random.choice(negative_list)
        negative_code = self.src_vec[negative_item]

        return anchor_code, positive_code, negative_code, anchor_label

class CodeDataSingle(Dataset):
    def __init__(self, src_vec, labels, train=True):
        self.is_train = train
        self.src_vec = src_vec
        self.labels = labels
        self.index = np.asarray([i for i in range(len(self.src_vec))])

    def __len__(self):
        return len(self.src_vec)

    def __getitem__(self, item):
        anchor_code = self.src_vec[item]
        anchor_label = self.labels[item]
        positive_list = [i for i in range(len(self.src_vec)) if self.labels[i] == anchor_label]
        positive_item = random.choice(positive_list)
        positive_code = self.src_vec[positive_item]

        negative_list = [i for i in range(len(self.src_vec)) if self.labels[i] != anchor_label]
        negative_item = random.choice(negative_list)
        negative_code = self.src_vec[negative_item]
        return anchor_code, positive_code, negative_code, anchor_label

class CodeDataAll(Dataset):
    def __init__(self, src_vec, ir_vec, byte_vec, labels, train=True):
        self.is_train = train
        self.src_vec = src_vec
        self.ir_vec = ir_vec
        self.byte_vec = byte_vec
        self.labels = labels
        self.index = np.asarray([i for i in range(len(self.src_vec))])

    def __len__(self):
        return len(self.src_vec)

    def __getitem__(self, item):
        # anchor--src pos--ir neg--other label src
        anchor_code = self.src_vec[item]
        anchor_label = self.labels[item]
        positive_code_1 = self.ir_vec[item]
        positive_code_2 = self.byte_vec[item]
        negative_list = [i for i in range(len(self.src_vec)) if self.labels[i] != anchor_label]
        negative_item = random.choice(negative_list)
        negative_code_1 = self.src_vec[negative_item]
        negative_code_2 = self.ir_vec[negative_item]
        negative_code_3 = self.byte_vec[negative_item]

        return anchor_code, positive_code_1, positive_code_2, negative_code_1, negative_code_2, negative_code_3, anchor_label

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

class TripletLossAll(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossAll, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive1: torch.Tensor, positive2: torch.Tensor, negative1: torch.Tensor, negative2: torch.Tensor,negative3: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive1) + self.calc_euclidean(anchor, positive2)
        distance_negative = self.calc_euclidean(anchor, negative1) + self.calc_euclidean(anchor, negative2) + self.calc_euclidean(anchor, negative3)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

class Network(nn.Module):
    def __init__(self, emb_dim, hidden_size, layers, dropout, embed_weight):
        super(Network, self).__init__()
        # self.hidden_size = hidden_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, 256, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=200-3+1),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        self.gru = nn.Sequential(
            nn.GRU(input_size=100, hidden_size=hidden_size, num_layers=layers, batch_first=True, dropout=dropout)
        )
        for p in self.gru.parameters():  # 正态分布的权值初始化
            nn.init.normal_(p, mean=0.0, std=0.001)
        # self.linear = nn.Linear(hidden_size, emb_dim)
        # self.embedding = nn.Embedding
        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embed_weight.to('cuda'), freeze=False).cuda()
        )
        # self.embedding.weight.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 200),
            # nn.Linear(emb_dim, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, emb_dim)
        )

    def forward(self, x):
        # out1 = self.embedding(torch.LongTensor(x).cuda())
        # out1 = self.embedding(x)
        # out2 = torch.LongTensor(trunc_and_pad(out1, self.voc_size, self.sentence_length))
        # out2 = self.fc(out1)

        x = self.embedding(x)
        out, hn = self.gru(x)
        # out = out.transpose(0, 1)
        # out = out.view(-1, self.hidden_size)
        out2 = self.fc(out)

        # out, hn = self.gru(x)
        # out = out.transpose(0, 1)
        # out = out.view(-1, self.hidden_size)
        # out = self.linear(out)
        # out = out.transpose(0, 1)

        # x = self.conv2(x)
        # x = x.view(-1, 64 * 4 * 4)
        # x = self.fc2(x)
        # x = nn.functional.normalize(x)
        return out2

class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, embed_dim, embed_weight):
        super(DPCNN, self).__init__()
        # self.config = config
        # self.channel_size = 250
        self.channel_size = 250
        self.embed_dim = embed_dim
        # self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, embed_dim), stride=1)
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, int(embed_dim/2)), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, embed_dim)
        # self.linear_out = nn.Linear(self.channel_size, 2)
        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embed_weight.to('cuda'), freeze=False).cuda()
        )
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        x = self.embedding(x)
        batch = x.shape[0]
        x = x.reshape(batch, 1, 200, 100)
        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]
        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        while x.size()[-2] > 2:
        # while x.size()[-2] > self.embed_dim:
            x = self._block(x)
        # x = x.view(batch, 2*self.channel_size)

        # x = x.view(batch, self.channel_size)

        x = x.reshape(batch, 250, -1)
        # x = self.linear_out(x)
        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

def process_all_data(pre_model_path, code, embed_arg, cls, doc_path, rank_file, K_fold, mul_bin_flag, retrain):
    # if code == "ir_id_1" or code == "byte_id_1":
    #     temp = code.split("_")[0]
    # else:
    #     temp = code
    temp = code
    pre_model_path = pre_model_path + temp + "_" + str(embed_arg['iter']) + "_" + str(
            embed_arg['window']) + "_" + str(embed_arg['voc_size']) + ".wordvectors"
    vec_path = "Word2vec/src_code/" + cls + "/" + temp + "_vec_" + str(embed_arg['iter']) + "_" + str(
        embed_arg['window']) + "_" + str(embed_arg['voc_size'])
    label_path = "Word2vec/src_code/" + cls + "/" + temp + "_label_" + str(embed_arg['iter']) + "_" + str(
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

def xgb(train_results, train_labels, val_results, val_labels, test_results, test_labels):
    train_results, val_results, test_results = train_results.reshape(len(train_results), -1), val_results.reshape(len(val_results), -1), test_results.reshape(len(test_results), -1)
    print("training XGBclassifier")
    tree = XGBClassifier(seed=2020)
    tree.fit(train_results, train_labels, eval_set=[(train_results,train_labels),(val_results, val_labels)], early_stopping_rounds=5)
    pred_labels = tree.predict(test_results)
    scores = get_score_binaryclassfication(pred_labels, test_labels)
    print(scores)

def rnn(train_results, train_labels, val_results, val_labels, test_results, test_labels, embed_arg, detect_arg, times, flag, classification_model, conn, save_base, embed_name):
    cherry_pick_new(train_results, train_labels, val_results, val_labels, test_results, test_labels,embed_arg, detect_arg,
                    times, flag, classification_model,conn, save_base, embed_name)

    return

def trunc_and_pad(vec, voc_size, sentence_length):
    if len(vec) >= sentence_length:
        vec = vec[:sentence_length]
    else:
        while len(vec) < sentence_length:
            vec.append([0]*voc_size)
    return vec

def generate_results(model, device, train_loader, val_loader, test_loader, categorical_flag, class_num):
    train_results = []
    train_labels = []
    model.eval()
    with torch.no_grad():
        for step, (data, _, _, label) in enumerate(train_loader):  # label??
            train_results.append(model(data.to(device)).cpu().numpy())
            # data_vec = model(data.to(device)).cpu().tolist()
            # train_results.append(np.asarray(trunc_and_pad(data_vec, sentence_length, voc_size)))
            train_labels.append(label)
    train_results = np.concatenate(train_results).astype(np.float16)
    train_labels = np.concatenate(train_labels)
    if categorical_flag == "True":
        train_labels = np_utils.to_categorical(train_labels, num_classes=class_num)

    print(train_results.shape)

    val_results, test_results = [], []
    val_labels, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for step, (data, _, _, label) in enumerate(val_loader):
            val_results.append(model(data.to(device)).cpu().numpy())
            # data_vec = model(data.to(device)).cpu().tolist()
            # val_results.append(np.asarray(trunc_and_pad(data_vec, sentence_length, voc_size)))
            val_labels.append(label)
        for step, (data, _, _, label) in enumerate(test_loader):
            test_results.append(model(data.to(device)).cpu().numpy())
            # data_vec = model(data.to(device)).cpu().tolist()
            # test_results.append(np.asarray(trunc_and_pad(data_vec, sentence_length, voc_size)))
            test_labels.append(label)

    val_results, val_labels = np.concatenate(val_results).astype(np.float16), np.concatenate(val_labels)
    test_results, test_labels = np.concatenate(test_results).astype(np.float16), np.concatenate(test_labels)
    return train_results, train_labels, val_results, val_labels, test_results, test_labels

def generate_results_all(model, device, train_loader, val_loader, test_loader, flag, class_num):
    train_results = []
    train_labels = []
    model.eval()
    with torch.no_grad():
        for step, (data, _, _, _, _, _, label) in enumerate(train_loader):  # label??
            train_results.append(model(data.to(device)).cpu().numpy())
            train_labels.append(label)
    train_results = np.concatenate(train_results).astype(np.float16)
    train_labels = np.concatenate(train_labels)
    if flag == 1:
        train_labels = np_utils.to_categorical(train_labels, num_classes=class_num)
    print(train_results.shape)

    val_results, test_results = [], []
    val_labels, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for step, (data, _, _, _, _, _, label) in enumerate(val_loader):
            val_results.append(model(data.to(device)).cpu().numpy())
            val_labels.append(label)
        for step, (data, _, _, _, _, _, label) in enumerate(test_loader):
            test_results.append(model(data.to(device)).cpu().numpy())
            test_labels.append(label)
    val_results, val_labels = np.concatenate(val_results).astype(np.float16), np.concatenate(val_labels)
    test_results, test_labels = np.concatenate(test_results).astype(np.float16), np.concatenate(test_labels)
    print(test_results.shape)
    return train_results, train_labels, val_results, val_labels, test_results, test_labels

def train_triplet_loss(model, embed_epoch, train_loader, cl_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.apply(init_weights)
    # model = torch.jit.script(model).to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.jit.script(TripletLoss())
    model.train()
    for epoch in range(embed_epoch):
        running_loss = []
        for step, (anchor_code, positive_code, negative_code, anchor_label) in enumerate(train_loader):
            anchor_code = anchor_code.to(device)
            positive_code = positive_code.to(device)
            negative_code = negative_code.to(device)

            # anchor_code = anchor_code.cuda()
            # positive_code = positive_code.cuda()
            # negative_code = negative_code.cuda()

            optimizer.zero_grad()
            anchor_out = model(anchor_code)
            positive_out = model(positive_code)
            negative_out = model(negative_code)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, embed_epoch, np.mean(running_loss)))
    torch.save(model.state_dict(), cl_model_path)
    return model

def get_index(code_doc, query_vocab, sentence_length, max_vocab):
    code_index = []
    for doc in code_doc:
        temp = [query_vocab[token] for token in doc.words]

        if len(temp) >= sentence_length:
            temp = temp[:sentence_length]
        else:
            while len(temp) < sentence_length:
                temp.append(max_vocab)
        code_index.append(temp)
    return code_index

def split_dataset_code(split_flag, categorical_flag, class_num, k, times, all_code_part, all_label_part):
    x_train, y_train = [], []
    if split_flag == "True":
        for i in range(1, k):
            # for i in range(0, k-1):
            if i - 1 == times:
                # if i == times:
                x_val = all_code_part[i]
                y_val = all_label_part[i]
            else:
                x_train += all_code_part[i]
                y_train += all_label_part[i]
        x_test, y_test = all_code_part[0], all_label_part[0]
    elif split_flag == "False":
        for i in range(k):
            if i == times:
                x_val = all_code_part[i]
                y_val = all_label_part[i]
            else:
                x_train += all_code_part[i]
                y_train += all_label_part[i]
        x_test, y_test = x_val, y_val

    # if categorical_flag == "False":
    #     y_train = np.asarray(y_train)
    # elif categorical_flag == "True":
    #     y_train = np_utils.to_categorical(np.asarray(y_train), num_classes=class_num)

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val), np.asarray(x_test), np.asarray(y_test)

def changeCode2Index(doc_path, rank_base, k, times, mul_bin_flag, split_flag, embed_model, categorical_flag, class_num, sentence_length, max_vocab):
    vocab = embed_model.wv.index2word
    # query_vocab = {v: k for k, v in vocab.items()}
    query_vocab = {v: k for k, v in enumerate(vocab)}
    all_code = split_codedoc(doc_path, rank_base, k, mul_bin_flag)
    all_label = [get_label(all_code[i], mul_bin_flag) for i in range(k)]
    all_index = [get_index(all_code[i], query_vocab, sentence_length, max_vocab) for i in range(k)]
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset_code(split_flag, categorical_flag, class_num, k, times, all_index, all_label)
    return x_train, y_train, x_val, y_val, x_test, y_test

def triplet_loss_2(doc_path, rank_base, class_num, times, K_fold, split_flag, categorical_flag, embed_arg, detect_arg, args, embed_name, classification_name,
                         mul_bin_flag, conn, save_base, enhance_embed_name, embed_base):
    cl_model_path = save_base + "/trained_model_"+enhance_embed_name+"_"+str(args["embed_epoch"])+"_"+str(args["hidden_size"])+"_"+str(args["layers"])+".pth"
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not torch.cuda.is_available():
        print("no gpu")
        exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name(device)
    
    embed_model = load_embed_model(embed_name, embed_base)
    all_vec = np.vstack((embed_model.vectors, np.zeros(embed_arg["voc_size"])))
    embed_weight = torch.FloatTensor(all_vec)
    # embed_weight = torch.FloatTensor(embed_model.vectors)
    max_vocab = len(embed_model.wv.index2word)
    x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = changeCode2Index(doc_path, rank_base, K_fold, times, mul_bin_flag, split_flag, embed_model, categorical_flag, class_num, embed_arg["sentence_length"], max_vocab)
    # if mul_bin_flag == 1:
    #     train_data, train_label = np.vstack((x_train_src, x_val_src)), np_utils.to_categorical(np.hstack((y_train_src, y_val_src)), num_classes=class_num)
    # else:

    train_data, train_label = np.vstack((x_train_src, x_val_src)), np.hstack((y_train_src, y_val_src))
    # train_ds, val_ds, test_ds = CodeDataSingle(np.asarray(train_data), np.asarray(train_label)), CodeDataSingle(np.asarray(x_val_src), np.asarray(y_val_src)), CodeDataSingle(np.asarray(x_test_src),np.asarray(y_test_src))
    train_ds, val_ds, test_ds = CodeDataSingle(train_data, train_label), CodeDataSingle(x_val_src, y_val_src), CodeDataSingle(x_test_src,y_test_src)

    train_loader = DataLoader(train_ds, batch_size=args["embed_batch_size"], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=args["embed_batch_size"], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=args["embed_batch_size"], shuffle=False, num_workers=8)

    # if enhance_embed_name == "mlp":
    #     model = Network(args["embed_dim"], args["hidden_size"], args["layers"], args["dropout"], embed_weight)
    if enhance_embed_name == "gru":
        model = Network(args["embed_dim"], args["hidden_size"], args["layers"], args["dropout"], embed_weight)
    elif enhance_embed_name == "dpcnn":
        model = DPCNN(args["embed_dim"], embed_weight)
    else:
        print("no embed model")
        exit(0)

    if os.path.exists(cl_model_path):
        print("loading...")
        model.load_state_dict(torch.load(cl_model_path))
        model = model.to(device)
    else:
        model = train_triplet_loss(model, args["embed_epoch"], train_loader, cl_model_path)
    del train_loader
    del train_ds
    if mul_bin_flag == 1:
        y_train_src = np_utils.to_categorical(np.asarray(y_train_src), num_classes=class_num)
    train_ds = CodeDataSingle(x_train_src, y_train_src)
    train_loader = DataLoader(train_ds, batch_size=args["embed_batch_size"], shuffle=True, num_workers=8)
    mlp_flag = 1
    # add mlp and remove mlp?
    train_results, train_labels, val_results, val_labels, test_results, test_labels = generate_results(model, device, train_loader, val_loader, test_loader, categorical_flag, class_num)
    rnn(train_results, train_labels, val_results, val_labels, test_results, test_labels, embed_arg, detect_arg, times, mul_bin_flag, classification_name, conn, save_base, embed_name)

# fasttext glove bert
# rnn > mlp and cnn
def triplet_loss_compare(all_src_vec_part, all_src_label_part, all_ir_vec_part, all_ir_label_part, all_byte_vec_part, all_byte_label_part,
                         class_num, times, K_fold, split_flag, categorical_flag, embed_arg, detect_arg, args, classification_model,
                         mul_bin_flag, conn, save_base, enhance_embed, embed):
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not torch.cuda.is_available():
        print("no gpu")
        exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name(device)
    x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = split_dataset(all_src_vec_part,
                                                                                           all_src_label_part,
                                                                                           class_num,
                                                                                           times, K_fold, split_flag,
                                                                                           categorical_flag)
    # pickle.dump(x_train_src, open(save_base+"/src_vec.pkl", "wb"))
    # pickle.dump(y_train_src, open(save_base+"/src_label.pkl", "wb"))
    if conn == "3loss_src":
        x_train_temp, y_train_temp = [],[]
        x_train_src, y_train_src = np.vstack((x_train_src, x_train_src)), np.hstack((y_train_src, y_train_src))
        # x_train_ir, y_train_ir, x_val_ir, y_val_ir, x_test_ir, y_test_ir = [], [], [], [], [], []
        # x_train_byte, y_train_byte, x_val_byte, y_val_byte, x_test_byte, y_test_byte = [], [], [], [], [], []
        train_ds, val_ds, test_ds = CodeDataSingle(x_train_src, y_train_src), CodeDataSingle(x_val_src, y_val_src), CodeDataSingle(x_test_src, y_test_src)
    elif conn == "3loss_ir":
        x_train_ir, y_train_ir, x_val_ir, y_val_ir, x_test_ir, y_test_ir = split_dataset(all_ir_vec_part,
                                                                                         all_ir_label_part,
                                                                                         class_num,
                                                                                         times, K_fold, split_flag,
                                                                                         categorical_flag)
        train_ds, val_ds, test_ds = CodeDataSingle(x_train_ir, y_train_ir), CodeDataSingle(x_val_ir,y_val_ir), CodeDataSingle(x_test_ir, y_test_ir)
    elif conn == "3loss_byte1" or conn == "3loss_byte2" or conn=="3loss_byte3":
        x_train_byte, y_train_byte, x_val_byte, y_val_byte, x_test_byte, y_test_byte = split_dataset(all_byte_vec_part,
                                                                                                     all_byte_label_part,
                                                                                                     class_num,
                                                                                                     times, K_fold,
                                                                                                     split_flag,
                                                                                                     categorical_flag)
        train_ds, val_ds, test_ds = CodeDataSingle(x_train_byte, y_train_byte), CodeDataSingle(x_val_byte,y_val_byte), CodeDataSingle(x_test_byte, y_test_byte)
    elif conn == "3loss_src_ir" or conn == "3loss_src_byte1" or conn=="3loss_src_byte2" or conn=="3loss_src_byte3":
        if conn == "3loss_src_ir":
            x_train_ir, y_train_ir, x_val_ir, y_val_ir, x_test_ir, y_test_ir = split_dataset(all_ir_vec_part, all_ir_label_part,
                                                                                             class_num,
                                                                                             times, K_fold, split_flag,
                                                                                             categorical_flag)
            x_train_byte, y_train_byte, x_val_byte, y_val_byte, x_test_byte, y_test_byte = [],[],[],[],[],[]
        elif conn == "3loss_src_byte1" or conn=="3loss_src_byte2" or conn=="3loss_src_byte3":
            x_train_ir, y_train_ir, x_val_ir, y_val_ir, x_test_ir, y_test_ir = [],[],[],[],[],[]
            x_train_byte, y_train_byte, x_val_byte, y_val_byte, x_test_byte, y_test_byte = split_dataset(all_byte_vec_part,
                                                                                                         all_byte_label_part,
                                                                                                         class_num,
                                                                                                         times, K_fold,
                                                                                                         split_flag,
                                                                                                         categorical_flag)
            x_train_byte, y_train_byte = np.vstack((x_train_byte, x_train_byte)), np.hstack((y_train_byte, y_train_byte))
        # train_ds, val_ds, test_ds = CodeData(x_train_src, x_train_ir, x_train_byte, y_train_src, conn), CodeData(x_val_src, x_val_ir, x_val_byte, y_val_src, conn),\
        #                             CodeData(x_test_src, x_test_ir, x_test_byte, y_test_src, conn)
        train_ds, val_ds, test_ds = CodeData(x_train_src, x_train_ir, x_train_byte, y_train_src, conn), CodeData(
            x_val_src, x_val_ir, x_val_byte, y_val_src, conn), \
                                    CodeData(x_test_src, x_test_ir, x_test_byte, y_test_src, conn)
    elif conn == "3loss_src_ir_byte1":
        x_train_ir, y_train_ir, x_val_ir, y_val_ir, x_test_ir, y_test_ir = split_dataset(all_ir_vec_part,
                                                                                         all_ir_label_part,
                                                                                         class_num,
                                                                                         times, K_fold, split_flag,
                                                                                         categorical_flag)
        x_train_byte, y_train_byte, x_val_byte, y_val_byte, x_test_byte, y_test_byte = split_dataset(all_byte_vec_part,
                                                                                                     all_byte_label_part,
                                                                                                     class_num,
                                                                                                     times, K_fold,
                                                                                                     split_flag,
                                                                                                     categorical_flag)
        train_ds, val_ds, test_ds = CodeDataAll(x_train_src, x_train_ir, x_train_byte, y_train_src), CodeDataAll(x_val_src, x_val_ir, x_val_byte, y_val_src), \
                                    CodeDataAll(x_test_src, x_test_ir, x_test_byte, y_test_src, conn)

    train_loader = DataLoader(train_ds, batch_size=args["embed_batch_size"], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=args["embed_batch_size"], shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=args["embed_batch_size"], shuffle=False, num_workers=8)
    # del x_train_src, y_train_src, x_train_ir, y_train_ir, x_train_byte, y_train_byte, train_ds, val_ds, test_ds
    # gc.collect()
    if enhance_embed == "gru":
        model = Network(args["embed_dim"], args["hidden_size"], args["layers"], args["dropout"])
    elif enhance_embed == "dpcnn":
        model = DPCNN(args["embed_dim"])
    else:
        print("no embed model")
        exit(0)
    model_path = save_base+"/trained_model.pth"
    if os.path.exists(model_path):
        print("loading...")
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    else:
        model.apply(init_weights)
        # model = torch.jit.script(model).to(device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if conn == "3loss_src_ir" or conn == "3loss_src_byte1" or conn == "3loss_src_byte2" or conn=="3loss_src_byte3" or conn == "3loss_src" or \
                conn == "3loss_ir" or conn == "3loss_byte1" or conn == "3loss_byte2" or conn=="3loss_byte3":
            criterion = torch.jit.script(TripletLoss())
            model.train()
            for epoch in range(args["embed_epoch"]):
                running_loss = []
                for step, (anchor_code, positive_code, negative_code, anchor_label) in enumerate(train_loader):
                    anchor_code = anchor_code.to(device)
                    positive_code = positive_code.to(device)
                    negative_code = negative_code.to(device)

                    optimizer.zero_grad()
                    anchor_out = model(anchor_code)
                    positive_out = model(positive_code)
                    negative_out = model(negative_code)

                    loss = criterion(anchor_out, positive_out, negative_out)
                    loss.backward()
                    optimizer.step()

                    running_loss.append(loss.cpu().detach().numpy())
                print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args["embed_epoch"], np.mean(running_loss)))

        elif conn == "3loss_src_ir_byte1":
            criterion = torch.jit.script(TripletLossAll())
            model.train()
            for epoch in range(args["embed_epoch"]):
                running_loss = []
                for step, (anchor_code, positive_code_1, positive_code_2, negative_code_1, negative_code_2, negative_code_3, anchor_label) in enumerate(train_loader):
                    anchor_code = anchor_code.to(device)
                    positive_code_1 = positive_code_1.to(device)
                    positive_code_2 = positive_code_2.to(device)
                    negative_code_1 = negative_code_1.to(device)
                    negative_code_2 = negative_code_2.to(device)
                    negative_code_3 = negative_code_3.to(device)

                    optimizer.zero_grad()
                    anchor_out = model(anchor_code)
                    positive_out_1 = model(positive_code_1)
                    positive_out_2 = model(positive_code_2)
                    negative_out_1 = model(negative_code_1)
                    negative_out_2 = model(negative_code_2)
                    negative_out_3 = model(negative_code_3)

                    loss = criterion(anchor_out, positive_out_1, positive_out_2, negative_out_1, negative_out_2, negative_out_3)
                    loss.backward()
                    optimizer.step()

                    running_loss.append(loss.cpu().detach().numpy())
                print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args["embed_epoch"], np.mean(running_loss)))

        # torch.save({"model_state_dict": model.state_dict(),
        #             "optimzier_state_dict": optimizer.state_dict()
        #             }, save_base+"/trained_model.pth")

        torch.save(model.state_dict(), save_base+"/trained_model.pth")
    if conn == "3loss_src_ir" or conn == "3loss_src_byte1" or conn == "3loss_src_byte2" or conn=="3loss_src_byte3" or conn == "3loss_src" or \
            conn == "3loss_ir" or conn == "3loss_byte1" or conn == "3loss_byte2" or conn=="3loss_byte3":
        train_results, train_labels, val_results, val_labels, test_results, test_labels = generate_results(model, device, train_loader, val_loader, test_loader, mul_bin_flag, class_num)
    elif conn == "3loss_src_ir_byte1" or conn == "3loss_src_ir_byte2":
        train_results, train_labels, val_results, val_labels, test_results, test_labels = generate_results_all(model,device,train_loader,val_loader,test_loader, mul_bin_flag, class_num)
    # del train_loader, test_loader
    # gc.collect()
    # if args.classifier == "xgb":
    #     xgb(train_results, train_labels, val_results, val_labels, test_results, test_labels)
    # elif args.classifier == "gru" or "lstm" or "bgru" or "blstm":
    rnn(train_results, train_labels, val_results, val_labels, test_results, test_labels, embed_arg, detect_arg, times, mul_bin_flag, classification_model, conn, save_base, embed)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.get_device_name(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', '-c')
    parser.add_argument('--code', '-d')
    # parser.add_argument('--hybrid_data', '-h')
    args = parser.parse_args()

    iter_range = [5]
    window_range = [5]
    sg_range = [0]
    min_count_range = [0]
    voc_size_range = [100]
    negative_range = [5]
    sample_range = [1e-3]
    hs_range = [0]
    sentence_length_range = [200]
    embed_batch_size = 32
    embed_dim = 128
    embed_epoch = 50
    hidden_size, layers, dropout = 64, 2, 0.3

    batch_size_range = [32, 128, 256]
    epochs_d_range = [40]
    lstm_unit_range = [32]
    optimizer_range = ["Adam"]
    layer_range = [2, 4]
    drop_out_range = [0.5]
    learning_rate_range = [0.0003, 0.001, 0.002]
    gru_unit_range = [128, 256, 512]
    dense_unit_range = [32, 64, 128]
    pool_size_range = [5, 10, 20]
    kernel_size_range = [5, 10, 20]

    times = 0
    cls = args.code

    base = "./"
    pre_model_path = base+"pre_train_def/spotbugs/w2v/"
    # src_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000/src/"
    # ir_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000/ir_id_1/"
    # byte_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000/byte_id_1/"
    # rank_file = base+"pickle_object/spotbugs/detect_bin_sample_15000/rank/"
    src_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000_test/src/"
    ir_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000_test/ir_id_1/"
    byte_doc_path = base+"pickle_object/spotbugs/detect_bin_sample_15000_test/byte_id_1/"
    rank_file = base+"pickle_object/spotbugs/detect_bin_sample_15000_test/rank/"

    K_fold = 11
    mul_bin_flag = 0
    class_num = 2
    retrain = "False"
    categorical_flag = "False"
    split_flag = "True"

    embed_arg = dict(min_count=min_count_range[0], voc_size=voc_size_range[0], sg=sg_range[0],
                     negative=negative_range[0], sample=sample_range[0], hs=hs_range[0],
                     iter=iter_range[0], window=window_range[0], sentence_length=sentence_length_range[0])
    detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epochs_d_range,
                      lstm_unit_range=lstm_unit_range, optimizer_range=optimizer_range,
                      layer_range=layer_range, drop_out_range=drop_out_range,
                      learning_rate_range = learning_rate_range,gru_unit_range=gru_unit_range,
                      dense_unit_range=dense_unit_range, pool_size_range=pool_size_range,
                      kernel_size_range=kernel_size_range)

    args = Merge(embed_arg, detect_arg)
    all_src_vec_part, all_src_label_part = process_all_data(pre_model_path, "src", embed_arg, cls, src_doc_path, rank_file, K_fold, mul_bin_flag, retrain)
    all_ir_vec_part, all_ir_label_part = process_all_data(pre_model_path, "ir_id_1", embed_arg, cls, ir_doc_path, rank_file, K_fold, mul_bin_flag, retrain)
    all_byte_vec_part, all_byte_label_part = process_all_data(pre_model_path, "byte_id_1", embed_arg, cls, byte_doc_path, rank_file, K_fold, mul_bin_flag, retrain)
    x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = split_dataset(all_src_vec_part,
                                                                                           all_src_label_part,
                                                                                           class_num,
                                                                                           times, K_fold, split_flag,
                                                                                           categorical_flag)
    x_train_ir, y_train_ir, x_val_ir, y_val_ir, x_test_ir, y_test_ir = split_dataset(all_ir_vec_part, all_ir_label_part,
                                                                                     class_num,
                                                                                     times, K_fold, split_flag,
                                                                                     categorical_flag)
    x_train_byte, y_train_byte, x_val_byte, y_val_byte, x_test_byte, y_test_byte = split_dataset(all_byte_vec_part,
                                                                                                 all_byte_label_part,
                                                                                                 class_num,
                                                                                                 times, K_fold,
                                                                                                 split_flag,
                                                                                                 categorical_flag)

    assert y_train_src.tolist()==y_train_ir.tolist()==y_train_byte.tolist()
    assert y_val_src.tolist()==y_val_ir.tolist()==y_val_byte.tolist()
    # x_train_byte, x_val_byte = [],[]
    train_ds = CodeData(x_train_src, x_train_ir, x_train_byte, y_train_src)
    train_loader = DataLoader(train_ds, batch_size=embed_batch_size, shuffle=True, num_workers=8)
    val_ds = CodeData(x_val_src, x_val_ir, x_val_byte, y_val_src)
    val_loader = DataLoader(val_ds, batch_size=embed_batch_size, shuffle=False, num_workers=8)
    test_ds = CodeData(x_test_src, x_test_ir, x_test_byte, y_test_src)
    test_loader = DataLoader(test_ds, batch_size=embed_batch_size, shuffle=False, num_workers=8)

    model = Network(embed_dim, hidden_size, layers, dropout)
    model.apply(init_weights)
    # model = torch.jit.script(model).to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.jit.script(TripletLoss())
    model.train()

    for epoch in range(embed_epoch):
        running_loss = []
        for step, (anchor_code, positive_code, negative_code, anchor_label) in enumerate(train_loader):
            anchor_code = anchor_code.to(device)
            positive_code = positive_code.to(device)
            negative_code = negative_code.to(device)

            optimizer.zero_grad()
            anchor_out = model(anchor_code, )
            positive_out = model(positive_code)
            negative_out = model(negative_code)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, embed_epoch, np.mean(running_loss)))

    torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
                }, "trained_model.pth")
    train_results = []
    train_labels = []
    model.eval()
    with torch.no_grad():
        # for code, _, _, label in tqdm(train_loader):
        for step, (code, _, _, label) in enumerate(train_loader):  # label??
            train_results.append(model(code.to(device)).cpu().numpy())
            train_labels.append(label)
    train_results = np.concatenate(train_results)
    train_labels = np.concatenate(train_labels)
    print(train_results.shape)

    val_results, test_results = [], []
    model.eval()
    with torch.no_grad():
        for step, (code, _, _, label) in enumerate(val_loader):
            val_results.append(model(code.to(device)).cpu().numpy())
        for step, (code, _, _, label) in enumerate(test_loader):
            test_results.append(model(code.to(device)).cpu().numpy())
    val_results = np.concatenate(val_results)
    test_results = np.concatenate(test_results)
    val_labels, test_labels = y_val_src, y_test_src
    print(test_results.shape)

    if args.classifier == "xgb":
        xgb(train_results, train_labels, val_results, val_labels, test_results, test_labels)
    elif args.classifier == "gru" or "lstm" or "bgru" or "blstm":
        rnn(train_results, train_labels, val_results, val_labels, test_results, test_labels)





