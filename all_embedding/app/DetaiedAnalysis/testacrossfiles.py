import pickle
import tensorflow as tf
import os
import random
import sys
sys.path.append("../../")
import collections
from all_embedding.app.appUtils import load_embed_model, get_vec_label, load_embed_arg, get_rank_score
from all_embedding.Hybrid.new.voting import group_vote, hard_vote, soft_vote
from generate_stat_csv import *
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

def get_sort_pattern(all_test_data, crl_base,_detect_model_base):
    labels = [item.cls for item in all_test_data]
    embed_name = ["w2v"]
    # embed_name = ["w2v", "fasttext", "glove", "elmo"]
    model_name = ['w2v_bgru']
    # model_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm',
    #               'fasttext_blstm', 'glove_bgru', 'fasttext_textcnn']
    n_bug = [5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 1500, 2000]
    pred_prob = []
    pred = []
    for i in range(len(embed_name)):
        embed_model = load_embed_model(embed_name[i], crl_base)
        embed_arg = load_embed_arg(embed_name[i])
        voc_size, sentence_length = embed_arg["voc_size"], embed_arg["sentence_length"]
        mul_bin_flag = 0
        all_vec, all_label = get_vec_label(embed_model, all_test_data, voc_size, sentence_length, mul_bin_flag)
        assert list(all_label) == labels
        for j in range(len(model_name)):
            if embed_name[i] in model_name[j]:
                detect_model = tf.keras.models.load_model(detect_model_base + model_name[j] + "/best_model_0.h5")
                y_pred_prob = detect_model.predict(all_vec)
                assert len(all_vec) == len(all_label) == len(all_test_data) == len(y_pred_prob.tolist())
                y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
                pred_prob.append(y_pred_prob)
                pred.append(y_pred)
            else:
                continue

    print("length of pred_prob", len(pred_prob))
    soft_vote_score, soft_prob = soft_vote(pred_prob, len(all_test_data), labels)
    hard_vote_score = hard_vote(pred, len(all_test_data), labels)
    soft_vote_score["top_n_tp"], soft_vote_score["top_n_precision"] = get_rank_score(soft_prob, labels, n_bug)

    rank_result = list(zip(soft_prob, all_test_data))
    rank_result.sort(reverse=True)
    ranked_data = [item[1] for item in rank_result]
    # ranked_prob, ranked_data = zip(*rank_result)

    original_sort_pattern = rank_pattern(all_test_data, "violation_type")
    part_sort_pattern = []
    for number in [1000, 2000, 4000, 6000, 8000, 10000, len(all_test_data)]:
        sort_pattern = rank_pattern(ranked_data[:number], "violation_type")
        part_sort_pattern.append(sort_pattern)
    return part_sort_pattern

def rank_pattern(dataset, keyword):
    pattern_dic = {}
    for item in dataset:
        pattern = item.info[keyword]
        if pattern not in pattern_dic.keys():
            pattern_dic[pattern] = 1
        else:
            pattern_dic[pattern] += 1
    return sorted(pattern_dic.items(), key = lambda x: x[1], reverse=True)




if __name__ == '__main__':
    doc_path = "../../app/data/src_top_35_invalid_commitIDs_removed_deduplication_by_token_multiple_label_issue_resolved_with_constructor_with_baseline"
    part_doc_path = "../../app/data/top21-35_invalid_commitIDs_removed_deduplication_by_token_multiple_label_issue_resolved_with_constructor_with_baseline"

    crl_base = "../../app/model/src_top_20/"
    detect_model_base = "../../app/score/"
    vote_save_base = "../../app/vote_score/cross_project/"

    if not os.path.exists(part_doc_path):
        all_code = pickle.load(open(doc_path, "rb"))
        top_20 = ['square-dagger', 'junit-team-junit', 'xetorthio-jedis', 'mrniko-redisson', 'checkstyle-checkstyle', 'swagger-api-swagger-core', 'swagger-api-swagger-codegen', 'alibaba-druid', 'dropwizard-metrics', 'google-error-prone', 'code4craft-webmagic', 'mybatis-mybatis-3', 'thinkaurelius-titan', 'google-auto', 'google-guava', 'Activiti-Activiti', 'mrniko-netty-socketio', 'clojure-clojure', 'spring-cloud-spring-cloud-netflix', 'netty-netty']
        top_35 = ['Activiti-Activiti', 'FasterXML-jackson-databind', 'alibaba-druid', 'mrniko-redisson', 'xetorthio-jedis', 'jankotek-MapDB', 'graphhopper-graphhopper', 'thinkaurelius-titan', 'jeremylong-DependencyCheck', 'zendesk-maxwell', 'google-guava', 'swagger-api-swagger-core', 'mrniko-netty-socketio', 'tananaev-traccar', 'code4craft-webmagic', 'junit-team-junit', 'droolsjbpm-optaplanner', 'dropwizard-metrics', 'sannies-mp4parser', 'clojure-clojure', 'google-auto', 'google-error-prone', 'square-dagger', 'mybatis-mybatis-3', 'spring-cloud-spring-cloud-netflix', 'netty-netty', 'bitcoinj-bitcoinj', 'immutables-immutables', 'OpenHFT-Chronicle-Queue', 'jitsi-jitsi-videobridge', 'swagger-api-swagger-codegen', 'square-wire', 'jboss-javassist-javassist', 'apache-hbase', 'checkstyle-checkstyle']
        top_15 = [name for name in top_35 if name not in top_20]
        all_code = [item for item in all_code if item.info["project_name"] in top_15]
        pickle.dump(all_code, open(part_doc_path, "wb"))
    else:
        all_code = pickle.load(open(part_doc_path, "rb"))

    all_test_data = []
    fix = [doc for doc in all_code if doc.cls == 1]
    unfix = [doc for doc in all_code if doc.cls == 0]
    random_list = random.sample(range(max(len(fix), len(unfix))), min(len(fix), len(unfix)))
    all_test_data.extend(fix)
    for i in random_list:
        all_test_data.append(unfix[i])

    labels = [item.cls for item in all_test_data]
    embed_name = ["w2v", "fasttext", "glove", "elmo"]
    # model_name = ['w2v_lstm']
    model_name = ['w2v_bgru', 'w2v_blstm', 'fasttext_bgru', 'elmo_bgru', 'w2v_textcnn', 'glove_blstm',
                       'fasttext_blstm', 'glove_bgru','fasttext_textcnn']
    n_bug = [5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 1500, 2000]
    pred_prob = []
    pred = []
    for i in range(len(embed_name)):
        embed_model = load_embed_model(embed_name[i], crl_base)
        embed_arg = load_embed_arg(embed_name[i])
        voc_size, sentence_length = embed_arg["voc_size"], embed_arg["sentence_length"]
        mul_bin_flag = 0
        all_vec, all_label = get_vec_label(embed_model, all_test_data, voc_size, sentence_length, mul_bin_flag)
        assert list(all_label) == labels
        for j in range(len(model_name)):
            if embed_name[i] in model_name[j]:
                detect_model = tf.keras.models.load_model(detect_model_base + model_name[j] + "/best_model_0.h5")
                y_pred_prob = detect_model.predict(all_vec)
                assert len(all_vec) == len(all_label) == len(all_test_data) == len(y_pred_prob.tolist())
                y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
                pred_prob.append(y_pred_prob)
                pred.append(y_pred)
            else:
                continue

    print("length of pred_prob", len(pred_prob))
    soft_vote_score, soft_prob = soft_vote(pred_prob, len(all_test_data), labels)
    hard_vote_score = hard_vote(pred, len(all_test_data), labels)
    soft_vote_score["top_n_tp"], soft_vote_score["top_n_precision"] = get_rank_score(soft_prob, labels, n_bug)

    rank_result = list(zip(soft_prob, all_test_data))
    rank_result.sort(reverse=True)
    ranked_data = [item[1] for item in rank_result]
    # ranked_prob, ranked_data = zip(*rank_result)

    numbers = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3000, 3356]
    original_sort_pattern = rank_pattern(all_test_data, "violation_type")
    part_sort_pattern = []
    for number in numbers:
        sort_pattern = rank_pattern(ranked_data[:number], "violation_type")
        part_sort_pattern.append(sort_pattern)
    pickle.dump(part_sort_pattern, open(vote_save_base+"sort_pattern.pkl", "wb"))
    generate_csv(part_sort_pattern, vote_save_base , numbers)
    print("done!")
