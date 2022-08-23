# from tensorflow.python.keras.layers import Conv1D
# from tensorflow.python.layers.convolutional import Conv1D
# from tensorflow.python.keras import Input
# from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization, CuDNNLSTM, CuDNNGRU
import random
import sys
sys.path.append("../../")
from Util.utils import *
from Util.gen_HYDE import *
from Bert.Bert_model import wrap_bert_model_run_detect
from app.appUtils import get_rank_score

def wrap_model_run_detect(x_train, y_train, x_val, y_val, args, flag, conn, classification_model):
    input_shape = get_input_shape(conn, args)
    if input_shape is None:
        exit(0)
    if classification_model == "lstm" or classification_model == "blstm" or classification_model == "gru" or classification_model == "bgru":
        detect_model = buildlstm_para_rnn(args, flag, classification_model, input_shape)
    elif classification_model == "lr":
        # detect_model = build_para_lr(args, flag, (250,))
        detect_model = build_para_lr(args, flag, input_shape)
    elif classification_model == "textcnn":
        detect_model = build_para_cnn(args, flag, input_shape)
    elif classification_model == "mlp":
        detect_model = build_para_mlp(args, flag, input_shape)
    # elif classification_model == "dpcnn":
    #     detect_mode = build_para_dpcnn()
    custom_early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0,
        mode='max'
    )
    history = detect_model.fit(x_train, y_train, batch_size=args["batch_size"], epochs=args["epochs_d"], validation_split=0.1 , callbacks=[custom_early_stopping])
    # history = detect_model.fit(x_train, y_train, batch_size=args["batch_size"], epochs=args["epochs_d"], validation_split=0.1)
    # history = detect_model.fit(x_train, y_train, batch_size=args["batch_size"], epochs=args["epochs_d"])
    if flag:
        y_pred = np.argmax(detect_model.predict(x_val), axis=-1)
        scores = get_score_multiclassfication(y_pred, y_val, args["class_num"])
    else:
        y_pred = (detect_model.predict(x_val) > 0.5).astype("int32")
        y_pred = y_pred.reshape(len(y_pred),)
        scores = get_score_binaryclassfication(y_pred, y_val)
    return scores, detect_model, history

def hybrid(k, max_length, voc_size, conn, all_vec):
    splice_row, splice_col = [],[]
    splice_add = []
    splice_sub = []
    splice_had = []
    splice_bin = []
    splice_all = []
    splice_all_cos, splice_all_euc = [],[]
    # all_src_vec, all_ir_vec, all_byte_vec = all_vec[0], all_vec[1], all_vec[2]
    for i in range(k):
        # if conn == "all_1":
        #     splice_row[i] = linear_vstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        #     splice_add[i] = add(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        #     splice_sub[i] = sub(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        #     splice_had[i] = mul(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        #     splice_bin[i] = linear_vstack(splice_add[i], splice_sub[i], splice_had[i])
        #     splice_all_cos[i] = Cosine_similarity_vstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        #     splice_all_euc[i] = Eucl_distance_vstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        #     splice_all[i] = linear_hstack(splice_row[i], splice_bin[i], splice_all_cos[i], splice_all_euc[i])
        # elif conn == "all_2":
        #     splice_col[i] = linear_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        #     splice_add[i] = add(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        #     splice_sub[i] = sub(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        #     splice_had[i] = mul(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        #     splice_bin[i] = linear_hstack(splice_add[i], splice_sub[i], splice_had[i])
        #     splice_all_cos[i] = Cosine_similarity_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        #     splice_all_euc[i] = Eucl_distance_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        #     splice_all[i] = linear_hstack(splice_col[i], splice_bin[i], splice_all_cos[i], splice_all_euc[i])
        # elif conn == "all_3":
        #     splice_all[i] = flatten_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], sentence_length_range,
        #                                    voc_size)
        if conn == "all_4":
            # splice_add.append(add(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size))
            # splice_sub.append(sub(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size))
            # splice_had.append(mul(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size))
            # splice_all_cos.append(Cosine_similarity_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i]))
            # splice_all_euc.append(Eucl_distance_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i]))
            # splice_all.append(linear_hstack(add(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size),
            #                                 sub(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size),
            #                                 mul(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size),
            #                                 Cosine_similarity_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i]),
            #                                Eucl_distance_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])))
            # splice_all.append(linear_hstack(sub(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size),
            #                                 Cosine_similarity_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])))
            all_vec[0][i] = linear_hstack(sub(all_vec[0][i], all_vec[1][i], all_vec[2][i], max_length, voc_size),
                                            Cosine_similarity_hstack(all_vec[0][i], all_vec[1][i], all_vec[2][i]))

            # del splice_add,splice_sub,splice_had,splice_all_cos,splice_all_euc
            # gc.collect()
        # elif conn == "linear_row":
        #     splice_all[i] = linear_vstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        # elif conn == "linear_col":
        #     splice_all[i] = linear_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
        # elif conn == "add":
        #     splice_all[i] = add(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        # elif conn == "sub":
        #     splice_all[i] = sub(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
        # elif conn == "mul":
        #     splice_all[i] = mul(all_src_vec[i], all_ir_vec[i], all_byte_vec[i], max_length, voc_size)
    # return splice_all
    return all_vec[0]


def cherry_pick(all_vec_part, all_label_part, embed_arg, detect_arg, times, split_flag, k, flag, classification_model, code, save_base):
    #########
    # embedding arg
    # sg_range = args_range["sg_range"]
    # min_count_range = args_range["min_count_range"]
    # voc_size_range = args_range["voc_size_range"]
    # negative_range = args_range["negative_range"]
    # sample_range = args_range["sample_range"]
    # hs_range = args_range["hs_range"]
    # iter_range = args_range["iter_range"]
    # window_range = args_range["window_range"]
    # sentence_length_range = args_range["sentence_length_range"]

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

    if flag:
        class_num = 9
    else:
        class_num = 2

    best_f1_score = -1
    best_args = {}
    best_score = {}
    check_exist()
    args1 = embed_arg
    categorical_flag = "False"
    print("start hybrid")
    if code in ["all_1", "all_2", "all_3", "all_4", "linear_row", "linear_col", "add", "sub", "mul"]:
        all_vec_part = hybrid(k, embed_arg["sentence_length"], embed_arg["voc_size"], code, all_vec_part)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(all_vec_part, all_label_part, class_num,
                                                                       times, k, split_flag, categorical_flag)
    print("success!")
    del all_vec_part, all_label_part
    gc.collect()
    # dump(x_train, y_train, 1000, save_base)
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
                                                args2 = dict(class_num=class_num, batch_size=batch_size, epochs_d=epochs_d,
                                                             lstm_unit=lstm_unit, optimizer=optimizer, layer=layer, drop_out=drop_out,
                                                             learning_rate = learning_rate, gru_unit=gru_unit, dense_unit=dense_unit,
                                                             pool_size=pool_size,kernel_size=kernel_size)
                                                args = Merge(args1, args2)
                                                score, detect_model, history = wrap_model_run_detect(x_train, y_train, x_val, y_val, args, flag, code, classification_model)
                                                write_record(args, score, save_base + "/tune_record_" + str(times))
                                                if flag:
                                                    f1 = score["micro_f1"]
                                                else:
                                                    f1 = score["f1"]
                                                if f1 > best_f1_score:
                                                    best_f1_score = f1
                                                    best_model = detect_model
                                                    best_args = args

                                                # else:
                                                #     del detect_model
                                                #     gc.collect()
    if flag:
        test_y_pre = np.argmax(best_model.predict(x_test), axis=-1)
        test_score = get_score_multiclassfication(test_y_pre, y_test, args["class_num"])
    else:
        test_y_pre = (best_model.predict(x_test) > 0.5).astype("int32").flatten()
        test_score = get_score_binaryclassfication(test_y_pre, y_test)
        # test_score = add_metric(best_model, x_test, y_test, test_score, times)

    # print("test score:", test_score)
    write_record(best_args, test_score, save_base + "/best_score_record")
    best_model.save(save_base + "/best_model_" + str(times) + ".h5")
    # plot_figure(best_history, data + "/" + code + "/", times)
    del best_model
    del x_train, y_train, x_val, y_val, x_test, y_test
    gc.collect()

def cherry_pick_new(x_train, y_train, x_val, y_val, x_test, y_test, embed_arg, detect_arg, times, flag, classification_model, conn, save_base, embed_model_name):
    #########
    # embedding arg
    # sg_range = args_range["sg_range"]
    # min_count_range = args_range["min_count_range"]
    # voc_size_range = args_range["voc_size_range"]
    # negative_range = args_range["negative_range"]
    # sample_range = args_range["sample_range"]
    # hs_range = args_range["hs_range"]
    # iter_range = args_range["iter_range"]
    # window_range = args_range["window_range"]
    # sentence_length_range = args_range["sentence_length_range"]

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

    if flag:
        class_num = 9
    else:
        class_num = 2

    best_f1_score = -1
    best_args = {}
    best_score = {}
    check_exist()
    args1 = embed_arg
    categorical_flag = "False"
    if embed_model_name == "bert_seq":
        for batch_size in batch_size_range:
            for epochs_d in epochs_d_range:
                for optimizer in optimizer_range:
                    for learning_rate in learning_rate_range:
                        args2 = dict(class_num=class_num, batch_size=batch_size, epochs_d=epochs_d,
                                     optimizer=optimizer, learning_rate=learning_rate)
                        args = Merge(args1, args2)
                        score, detect_model, history = wrap_bert_model_run_detect(x_train, y_train, x_val, y_val, args,
                                                                                  flag, conn, classification_model,
                                                                                  embed_model_name)
                        if flag:
                            f1 = score["micro_f1"]
                        else:
                            f1 = score["f1"]
                        if f1 > best_f1_score:
                            best_f1_score = f1
                            best_model = detect_model
                        else:
                            del detect_model
                            gc.collect()
    else:
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
                                                    args2 = dict(class_num=class_num, batch_size=batch_size, epochs_d=epochs_d,
                                                                 lstm_unit=lstm_unit, optimizer=optimizer, layer=layer, drop_out=drop_out,
                                                                 learning_rate = learning_rate, gru_unit=gru_unit, dense_unit=dense_unit,
                                                                 pool_size=pool_size, kernel_size=kernel_size)
                                                    args = Merge(args1, args2)
                                                    score, detect_model, history = wrap_model_run_detect(x_train, y_train, x_val, y_val, args, flag, conn, classification_model)
                                                    write_record(args, score, save_base + "/tune_record_" + str(times))
                                                    if flag:
                                                        f1 = score["micro_f1"]
                                                    else:
                                                        f1 = score["f1"]
                                                    if f1 > best_f1_score:
                                                        best_f1_score = f1
                                                        best_model = detect_model
                                                        best_args = args
                                                    # else:
                                                    #     del detect_model
                                                    #     gc.collect()
    if flag:
        test_y_pre = np.argmax(best_model.predict(x_test), axis=-1)
        test_score = get_score_multiclassfication(test_y_pre, y_test, args["class_num"])
    else:
        test_y_pre = (best_model.predict(x_test) > 0.5).astype("int32").flatten()
        # y_prob = best_model.predict(x_test)
        test_score = get_score_binaryclassfication(test_y_pre, y_test)
        # test_score = add_metric(best_model, x_test, y_test, test_score, times)
    # print("test score:", test_score)
    # test_score["top_n_tp"], test_score["top_n_precision"] = get_rank_score(y_prob, y_test.tolist(), n_bug)
    write_record(best_args, test_score, save_base + "/best_score_record")
    if embed_model_name == "bert_seq":
        if not os.path.isdir(save_base+"/"+"best_model_"+ str(times)):
            os.mkdir(save_base+"/"+"best_model_"+ str(times))
        best_model.save_weights(save_base + "/best_model_" + str(times) + "/best_model")
    else:
        best_model.save(save_base + "/best_model_" + str(times) + ".h5")
    pickle.dump([x_test, y_test], open(save_base+"testdata_" + str(time), "wb"))
    del x_train, y_train, x_val, y_val, x_test, y_test
    gc.collect()

    return best_model, args, test_score, test_y_pre