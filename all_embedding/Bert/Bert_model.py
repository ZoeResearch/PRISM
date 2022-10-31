import pickle
import random
import pickle
from Bert.utils import *
# from Util.training import *
from Util.utils import *
import transformers


def create_finetune_model(n_out, max_length, model_name, learning_rate, flag, optimizer, layers, bert_model):
    if bert_model == "bert_seq":
        MAX_SEQ_LEN = max_length
        BERT_NAME = model_name
        bert = TFBertModel.from_pretrained(BERT_NAME)
        input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
        input_type = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
        input_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
        # inputs = [input_ids, input_type, input_mask]
        # bert_outputs = bert(inputs)
        # last_hidden_states = bert_outputs.last_hidden_state
        # avg = GlobalAveragePooling1D()(last_hidden_states)

        x = bert(input_ids, attention_mask=input_mask, token_type_ids=input_type)
        x = x.last_hidden_state

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        for i in range(layers):
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)

        if flag == 1:
            # output = Dense(n_out, activation="softmax")(avg)
            x = tf.keras.layers.Dense(n_out, activation='softmax')(x)
        elif flag == 0:
            # output = Dense(1, activation="sigmoid")(avg)
            x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # model = keras.Model(inputs=inputs, outputs=output)
        model = tf.keras.Model(inputs=[input_ids, input_type, input_mask], outputs=x)
    elif bert_model == "codebert_seq" or bert_model == "roberta_seq":
        MAX_SEQ_LEN = max_length + 2
        BERT_NAME = model_name
        roberta_model = TFRobertaModel.from_pretrained(BERT_NAME)

        input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
        input_type = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
        input_mask = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
        # inputs = [input_ids, input_type, input_mask]

        x = roberta_model(input_ids, attention_mask = input_mask, token_type_ids = input_type)

        x = x[0]
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

        for i in range(layers):
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)

        if flag == 1:
            x = tf.keras.layers.Dense(n_out, activation='softmax')(x)
        else:
            x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=[input_ids, input_type, input_mask], outputs=x)

    if flag:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.adam_v2.Adam(lr=learning_rate),
                      # optimizer=optimizer,
                      metrics=['accuracy'])

    else:
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.adam_v2.Adam(lr=learning_rate),
                      # optimizer=optimizer,
                      metrics=["accuracy"])

    model.summary()
    return model

def build_detect_model(bert_model, args, conn, classification_model, flag):
    if bert_model == "bert_seq" or bert_model == "codebert_seq" or bert_model == "roberta_seq":
        detect_model = create_finetune_model(args["class_num"], args["max_length"], args["model_name"], args["learning_rate"], flag, args["optimizer"],  args["encode_layers"], bert_model)
    elif bert_model == "bert_token" or bert_model == "codebert_token" or bert_model == "roberta_token":
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
        patience=10,
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
    if bert_model == "bert_seq" or bert_model == "codebert_seq" or bert_model == "roberta_seq":
        ##########
        # detect arg
        batch_size_range = detect_arg["batch_size_range"]
        epochs_d_range = detect_arg["epochs_d_range"]
        optimizer_range = detect_arg["optimizer_range"]
        learning_rate_range = detect_arg["learning_rate_range"]
        encode_layer_range = detect_arg["encode_layer_range"]


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

        for batch_size in batch_size_range:
            for epochs_d in epochs_d_range:
                for optimizer in optimizer_range:
                    for learning_rate in learning_rate_range:
                        for encode_layers in encode_layer_range:
                            args2 = dict(class_num=class_num, batch_size=batch_size, epochs_d=epochs_d,
                                         optimizer=optimizer, learning_rate = learning_rate, encode_layers = encode_layers)
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
                                # best_history = history
                                # print("better f1:", f1)
                                # print("better args:", best_args)
                            else:
                                del detect_model
                                gc.collect()
    elif bert_model == "bert_token" or bert_model == "codebert_token" or bert_model == "roberta_token":
        ##########
        # detect arg
        batch_size_range = detect_arg["batch_size_range"]
        epochs_d_range = detect_arg["epochs_d_range"]
        lstm_unit_range = detect_arg["lstm_unit_range"]
        optimizer_range = detect_arg["optimizer_range"]
        layer_range = detect_arg["layer_range"]
        drop_out_range = detect_arg["drop_out_range"]
        learning_rate_range = detect_arg["learning_rate_range"]
        gru_unit_range = detect_arg["gru_unit_range"]
        dense_unit_range = detect_arg["dense_unit_range"]
        pool_size_range = detect_arg["pool_size_range"]
        kernel_size_range = detect_arg["kernel_size_range"]

        if classification_model == "lstm" or classification_model == "blstm":
            gru_unit_range = [gru_unit_range[0]]
            dense_unit_range = [dense_unit_range[0]]
            pool_size_range = [pool_size_range[0]]
            kernel_size_range = [kernel_size_range[0]]
        elif classification_model == "gru" or classification_model == "bgru":
            lstm_unit_range = [lstm_unit_range[0]]
            dense_unit_range = [dense_unit_range[0]]
            pool_size_range = [pool_size_range[0]]
            kernel_size_range = [kernel_size_range[0]]
        elif classification_model == "mlp":
            gru_unit_range = [gru_unit_range[0]]
            lstm_unit_range = [lstm_unit_range[0]]
            pool_size_range = [pool_size_range[0]]
            kernel_size_range = [kernel_size_range[0]]
        elif classification_model == "textcnn":
            gru_unit_range = [gru_unit_range[0]]
            lstm_unit_range = [lstm_unit_range[0]]
            dense_unit_range = [dense_unit_range[0]]

        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(all_input, all_label_part, class_num,
                                                                       times, k, split_flag, categorical_flag)
        del all_input, all_label_part
        gc.collect()
        for batch_size in batch_size_range:
            for epochs_d in epochs_d_range:
                for lstm_unit in lstm_unit_range:
                    for optimizer in optimizer_range:
                        for layer in layer_range:
                            for drop_out in drop_out_range:
                                for learning_rate in learning_rate_range:
                                    for gru_unit in gru_unit_range:
                                        for dense_unit in dense_unit_range:
                                            for pool_size in pool_size_range:
                                                for kernel_size in kernel_size_range:
                                                    args2 = dict(class_num=class_num, batch_size=batch_size,
                                                                 epochs_d=epochs_d,
                                                                 lstm_unit=lstm_unit, optimizer=optimizer, layer=layer,
                                                                 drop_out=drop_out,
                                                                 learning_rate=learning_rate, gru_unit=gru_unit,
                                                                 dense_unit=dense_unit,
                                                                 pool_size=pool_size, kernel_size=kernel_size)
                                                    args = Merge(args1, args2)
                                                    score, detect_model, history = wrap_bert_model_run_detect(x_train, y_train,
                                                                                                         x_val, y_val, args,
                                                                                                         flag, code,
                                                                                                         classification_model, bert_model)
                                                    write_record(args, score, save_base + "/tune_record_" + str(times))
                                                    if flag:
                                                        f1 = score["micro_f1"]
                                                    else:
                                                        f1 = score["f1"]
                                                    if f1 > best_f1_score:
                                                        best_f1_score = f1
                                                        best_model = detect_model
                                                        best_args = args
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
    if bert_model == "bert_seq" or bert_model == "codebert_seq" or bert_model == "roberta_seq":
        if not os.path.isdir(save_base+"/"+"best_model_"+ str(times)):
            os.mkdir(save_base+"/"+"best_model_"+ str(times))
        best_model.save_weights(save_base+"/"+"best_model_"+ str(times)+"/best_model")

    # test =  create_model(args["class_num"], args["max_length"], args["model_name"], args["learning_rate"], code, classification_model)
    # test.load_weights(save_base + "/best_model_" + str(times))
    # print("pass test!")

    del best_model
    del x_train, y_train, x_val, y_val, x_test, y_test
    gc.collect()



