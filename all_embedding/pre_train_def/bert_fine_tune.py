import collections
import pickle
import os
from transformers import BertTokenizer
from transformers import TFBertModel
import keras
from keras.layers import Input
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
import tensorflow as tf
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls')


def convert_example_to_feature(review, tokenizer, max_length):
    # 单个数据转换为bert的输入形式 encode_plue返回三个元素的字典
    # max_length = 200
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to BERT
                                 pad_to_max_length = True,  # add [PAD] tokens  padding='longest'
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True
                                 )


def encode_examples(raw_code, label, name, max_length):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    #输入：原始代码token list， label
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    tokenizer = BertTokenizer.from_pretrained(name)

    # for codeDoc in raw_code:
    for i in range(len(raw_code)):
        bert_input = convert_example_to_feature(raw_code[i], tokenizer, max_length)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(int(label[i]))
    print("length of bert_input", len(input_ids_list))
    return [input_ids_list, token_type_ids_list, attention_mask_list], label_list

def read_file(doc_path):
    f = open(doc_path, "rb")
    all_code_blocks = []
    all_labels = []
    for block in pickle.load(f):
        if block.words:
            code = list(filter(None, [i.replace("\n", "") for i in block.words]))
            all_code_blocks.append(code)
            all_labels.append(block.cls)
    return all_code_blocks, all_labels

def split(X, y, rank, class_num):
    all_data_len = len(X)
    X_after_shuffle = []
    y_after_shuffle = []
    for i in rank:
        X_after_shuffle.append(X[i])
        y_after_shuffle.append(y[i])
    num_train = int(all_data_len // 1.25)
    X_after_shuffle = np.array(X_after_shuffle)
    y_after_shuffle = np.array(y_after_shuffle)
    x_train, x_test = X_after_shuffle[:num_train], X_after_shuffle[num_train:]
    y_train, y_test = y_after_shuffle[:num_train], y_after_shuffle[num_train:]
    train_val_len = len(x_train)
    num_train_val = int(train_val_len * 0.75)
    x_train, x_val = x_train[:num_train_val], x_train[num_train_val:]
    y_train, y_val = y_train[:num_train_val], y_train[num_train_val:]
    y_train = np_utils.to_categorical(y_train, num_classes=class_num)
    return x_train, x_val, x_test, y_train, y_val, y_test

def makedata(ori_vec, y, class_num):
    all_data_len = len(ori_vec[0])
    rank_filename = "../../pickle_object/rank" + str(all_data_len)
    if os.path.exists(rank_filename):
        rank_file = open(rank_filename, "rb")
        rank = pickle.load(rank_file)
    else:
        rank_file = open(rank_filename, "wb")
        rank = [i for i in range(all_data_len)]
        random.shuffle(rank)
        pickle.dump(rank, rank_file)

    X_train_com = []
    X_val_com = []
    X_test_com = []
    for i in range(len(ori_vec)):  #ori是三个list组成的list
        x_train, x_val, x_test, y_train, y_val, y_test = split(ori_vec[i], y, rank, class_num)
        X_train_com.append(x_train)
        X_val_com.append(x_val)
        X_test_com.append(x_test)
    return X_train_com, X_val_com, X_test_com, y_train, y_val, y_test
def create_model(n_out, max_length, model_name):
    MAX_SEQ_LEN = max_length
    BERT_NAME = model_name
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_type = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    input_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]
    bert = TFBertModel.from_pretrained(BERT_NAME)
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = GlobalAveragePooling1D()(last_hidden_states)
    output = Dense(n_out, activation="softmax")(avg)
    # output = Dense(n_out, activation="softmax")(avg)
    model = keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model

def wrap_model(train_X, train_y, test_X, test_y, class_num, max_length, model_name, epoch, batch_size, save_path)):
    detect_model = create_model(class_num, args["max_length"], args["model_name"])
    detect_model.compile(loss='categorical_crossentropy',
                         optimizer=args["optimizer"],
                         metrics=['accuracy'])

    detect_model.fit(train_X, train_y, epochs=args["epochs"], batch_size=args["batch_size"],
                     callbacks=[TensorBoard(log_dir='./tmp/log')])
    y_pred = np.argmax(detect_model.predict(val_X, batch_size=args["batch_size"]), axis=-1)
    detect_model.save_pretrained(model_path)

if __name__ == '__main__':
    src_path = "../pickle_object/sard/embedding/doc_src_embedding"
    ir_path = "../pickle_object/sard/embedding/doc_IR_embedding"
    byte_path = "../pickle_object/sard/embedding/doc_byte_embedding"
    src_model_path = "./sard/bert/bert_src"
    ir_model_path = "./sard/bert/bert_ir"
    byte_model_path = "./sard/bert/bert_byte"
    model_name = ""
    src_max_length = 200
    class_num = 2
    epoch = [20, 30]
    batch_size = [32, 64, 128]


    print("train src...")
    doc_path, save_path, max_length = src_path, src_model_path, src_max_length
    code_blocks, code_label = read_file(doc_path)
    print(code_blocks[0])

    if os.path.isfile(save_path):
        os.remove(save_path)

    ds_encoded, label = encode_examples(code_blocks, code_label, model_name, max_length)
    train_X, test_X, train_y, test_y = makedata(ds_encoded, label, class_num)
    wrap_model(train_X, train_y, test_X, test_y, class_num, max_length, model_name, epoch, batch_size, save_path)

