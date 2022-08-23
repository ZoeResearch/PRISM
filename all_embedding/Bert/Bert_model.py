import pickle
import random
import pickle
from Bert.utils import *
# from Util.training import *
from Util.utils import *
import transformers

def create_finetune_model(n_out, max_length, model_name, learning_rate, flag):
    MAX_SEQ_LEN = max_length
    BERT_NAME = model_name
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    input_type = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    input_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    inputs = [input_ids, input_type, input_mask]
    bert = TFBertModel.from_pretrained("microsoft/codebert-base")
    bert_outputs = bert(inputs)
    last_hidden_states = bert_outputs.last_hidden_state
    avg = GlobalAveragePooling1D()(last_hidden_states)
    if flag == 1:
        output = Dense(n_out, activation="softmax")(avg)
    elif flag == 0:
        output = Dense(1, activation="sigmoid")(avg)
    model = keras.Model(inputs=inputs, outputs=output)
    if flag:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.adam_v2.Adam(lr=learning_rate),
                      metrics=['accuracy'])

    else:
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.adam_v2.Adam(lr=learning_rate),
                      metrics=["accuracy"])
    model.summary()
    return model

def build_detect_model(bert_model, args, conn, classification_model, flag):
    if bert_model == "bert_seq" or bert_model == "codebert_seq":
        detect_model = create_finetune_model(args["class_num"], args["max_length"], args["model_name"], args["learning_rate"], flag)
    elif bert_model == "bert_token" or bert_model == "codebert_token":
        input_shape = get_input_shape(conn, args)
        if input_shape is None:
            print("no input_shape")
            exit(0)
        if classification_model == "lstm" or classification_model == "blstm" or classification_model == "gru" or classification_model == "bgru":
            detect_model = buildlstm_para_rnn(args, flag, classification_model, input_shape)
        elif classification_model == "lr":
            detect_model = build_para_lr(args, flag, (250,))
        elif classification_model == "textcnn":
            detect_model = build_para_cnn(args, flag, input_shape)
        elif classification_model == "mlp":
            detect_model = build_para_mlp(args, flag, input_shape)
    return detect_model

def wrap_bert_model_run_detect(x_train, y_train, x_val, y_val, args, flag, conn, classification_model, bert_model):

    detect_model = build_detect_model(bert_model, args, conn, classification_model, flag)
    custom_early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0,
        mode='max'
    )
    history = detect_model.fit(x_train, y_train, epochs=args["epochs_d"], batch_size=args["batch_size"], validation_split=0.1 , callbacks=[custom_early_stopping])

    if flag:
        y_pred = np.argmax(detect_model.predict(x_val, batch_size=args["batch_size"]), axis=-1)
        scores = get_score_multiclassfication(y_pred, y_val, args["class_num"])
    else:
        y_pred = (detect_model.predict(x_val) > 0.5).astype("int32")
        scores = get_score_binaryclassfication(y_pred, y_val)

    return scores, detect_model, history

def cherry_pick(all_input, all_label_part, embed_arg, detect_arg, times, split_flag, k, flag, classification_model, code, save_base, bert_model):
    ##########
    # detect arg
    batch_size_range = detect_arg["batch_size_range"]
    epochs_d_range = detect_arg["epochs_d_range"]
    optimizer_range = detect_arg["optimizer_range"]
    learning_rate_range = detect_arg["learning_rate_range"]

    if flag:
        class_num = 9
    else:
        class_num = 2

    best_f1_score = -1
    best_args = {}
    best_score = {}
    check_exist()
    args1 = embed_arg
    categorical_flag = "True"
    if bert_model == "bert_seq":
        all_input_ids, all_token_type_ids, all_attention_mask = all_input[0], all_input[1], all_input[2]
        x_train_input_id, y_train_input_id, x_val_input_id, y_val_input_id, x_test_input_id, y_test_input_id = split_dataset(all_input_ids, all_label_part, class_num,
                                                                           times, k, split_flag, categorical_flag)
        x_train_type_id, y_train_type_id, x_val_type_id, y_val_type_id, x_test_type_id, y_test_type_id = split_dataset(all_token_type_ids, all_label_part, class_num,
            times, k, split_flag, categorical_flag)
        x_train_mask, y_train_mask, x_val_mask, y_val_mask, x_test_mask, y_test_mask = split_dataset(all_attention_mask, all_label_part, class_num,
            times, k, split_flag, categorical_flag)
        x_train = [x_train_input_id, x_train_type_id, x_train_mask]
        x_val = [x_val_input_id, x_val_type_id, x_val_mask]
        x_test = [x_test_input_id, x_test_type_id, x_test_mask]
        y_train, y_val, y_test = y_train_input_id, y_val_input_id, y_test_input_id
    elif bert_model == "codebert_token":
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(all_input, all_label_part, class_num,
                                                                       times, k, split_flag, categorical_flag)

    for batch_size in batch_size_range:
        for epochs_d in epochs_d_range:
            for optimizer in optimizer_range:
                for learning_rate in learning_rate_range:
                    args2 = dict(class_num=class_num, batch_size=batch_size, epochs_d=epochs_d,
                                 optimizer=optimizer, learning_rate = learning_rate)
                    # print("training args:", args2)
                    args = Merge(args1, args2)
                    score, detect_model, history = wrap_bert_model_run_detect(x_train, y_train, x_val, y_val, args, flag, code, classification_model, bert_model)
                    # print("val score:", score)
                    write_record(args, score, save_base + "/tune_record_" + str(times))
                    if flag:
                        f1 = score["micro_f1"]
                    else:
                        f1 = score["f1"]
                    if f1 > best_f1_score:
                        best_f1_score = f1
                        best_model = detect_model
                        best_args = args

                    else:
                        del detect_model
                        gc.collect()

    if flag:
        test_y_pre = np.argmax(best_model.predict(x_test), axis=-1)
        test_score = get_score_multiclassfication(test_y_pre, y_test, args["class_num"])
    else:
        test_y_pre = (best_model.predict(x_test) > 0.5).astype("int32").flatten()
        test_score = get_score_binaryclassfication(test_y_pre, y_test)
    print("test score:", test_score)
    write_record(best_args, test_score, save_base + "/best_score_record")
    if bert_model == "bert_seq" or bert_model == "codebert_seq":
        if not os.path.isdir(save_base+"/"+"best_model_"+ str(times)):
            os.mkdir(save_base+"/"+"best_model_"+ str(times))
        best_model.save_weights(save_base+"/"+"best_model_"+ str(times)+"/best_model")

    del best_model
    del x_train, y_train, x_val, y_val, x_test, y_test
    gc.collect()



