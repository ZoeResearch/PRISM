import collections
import pickle
import random
import statistics
import warnings
import sys
import os

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')


def load_pickle(pickle_path):
    file = open(pickle_path, 'rb')
    file_content = pickle.load(file)
    file.close()

    fixed_train = 0
    unfixed_train = 0
    fixed_test = 0
    unfixed_test = 0
    train_all_ = 0
    test_all_ = 0
    project_list_for_training = ["Activiti-Activiti","code4craft-webmagic","junit-team-junit","spring-cloud-spring-cloud-netflix","xetorthio-jedis","alibaba-druid","dropwizard-metrics","mrniko-netty-socketio","square-dagger","google-auto","mrniko-redisson","swagger-api-swagger-codegen","checkstyle-checkstyle","google-error-prone","mybatis-mybatis-3","swagger-api-swagger-core","clojure-clojure","google-guava","netty-netty","thinkaurelius-titan"]

    for _ in file_content:
        # train
        if _[3]["project_name"] in project_list_for_training:
            if _[2] == 1:
                fixed_train += 1
            else:
                unfixed_train += 1
            train_all_ += 1
        else:  # test
            if _[2] == 1:
                fixed_test += 1
            else:
                unfixed_test += 1
            test_all_ += 1
    print("[+] Statistics:")
    print(" # Train Fixed Instances:", fixed_train)
    print(" # Train Unfixed Instances:", unfixed_train)
    print(" # Train All Instances:", train_all_)
    print(" # Test Fixed Instances:", fixed_test)
    print(" # Test Unfixed Instances:", unfixed_test)
    print(" # Test All Instances:", test_all_)

    # For Training
    Label = []
    F21 = []
    F20 = []
    F22 = []
    F117 = []
    F110 = []
    F115 = []

    # For Testing
    Label_test = []
    F21_test = []
    F20_test = []
    F22_test = []
    F117_test = []
    F110_test = []
    F115_test = []
    random.shuffle(file_content)
    train_unfixed_count = 0
    test_unfixed_count = 0
    progress = 0
    for ind, codedocument in enumerate(file_content):
        if test_unfixed_count >= fixed_test and codedocument[2] == 0 and codedocument[3]["project_name"] not in project_list_for_training:
            continue
        if train_unfixed_count >= fixed_train and codedocument[2] == 0 and codedocument[3]["project_name"] in project_list_for_training:
            continue
        words = codedocument[0]
        tags = codedocument[1]
        cls = codedocument[2]  # cls=0: unfixed(unactionable); cls=1: fixed(actionable)
        info = codedocument[3]
        if cls == 0:  # unfixed
            if info["project_name"] in project_list_for_training:
                train_unfixed_count += 1
            else:
                test_unfixed_count += 1
        # F21  | warning type                          | info["warning_category"]
        # F20  | warning pattern                       | info["violation_type"]
        # F22  | warning priority                      | info["priority"]
        # F117 | defect likelihood for warning pattern | info["defect_likelihood_for_warning_pattern"]
        # F110 | warning context for warning type      | info["warning_context_for_warning_type"]
        # F115 | warning context in file               | info["warning_context_of_file"]

        # [Testing Set]
        if info["project_name"] not in project_list_for_training:
            F21_test.append(info["warning_category"])
            F20_test.append(info["violation_type"])
            F22_test.append(info["priority"])
            F117_test.append(info["defect_likelihood_for_warning_pattern"])
            F110_test.append(info["warning_context_for_warning_type"])
            F115_test.append(info["warning_context_of_file"])
            Label_test.append(cls)
        else:  # [Training Set]
            F21.append(info["warning_category"])
            F20.append(info["violation_type"])
            F22.append(info["priority"])
            F117.append(info["defect_likelihood_for_warning_pattern"])
            F110.append(info["warning_context_for_warning_type"])
            F115.append(info["warning_context_of_file"])
            Label.append(cls)
        progress += 1
        # print(progress)
    # initialize data of lists.
    data_train = {
        'F21': F21,
        'F20': F20,
        'F22': F22,
        'F117': F117,
        'F110': F110,
        'F115': F115
        }

    data_test = {
        'F21': F21_test,
        'F20': F20_test,
        'F22': F22_test,
        'F117': F117_test,
        'F110': F110_test,
        'F115': F115_test
    }
    # Create DataFrame
    training_x = pd.DataFrame(data_train)
    training_y = Label
    testset_x = pd.DataFrame(data_test)
    testset_y = Label_test
    return training_x, training_y, testset_x, testset_y


def main(path, stop_at, clf, interesting_path, seed=0):
    if not os.path.exists(path):
        os.makedirs(path)
    training_x, training_y, testset_x, testset_y = load_pickle("data/src_top_35_invalid_commitIDs_removed_deduplication_by_token_multiple_label_issue_resolved_with_constructor_with_baseline")

    print("training_x\n", training_x.columns)

    label_encoder = LabelEncoder()
    training_x['F20'] = label_encoder.fit_transform(training_x.F20.values)
    training_x['F21'] = label_encoder.fit_transform(training_x.F21.values)

    testset_x['F20'] = label_encoder.fit_transform(testset_x.F20.values)
    testset_x['F21'] = label_encoder.fit_transform(testset_x.F21.values)

    scaler = MinMaxScaler()
    testset_x = pd.DataFrame(scaler.fit_transform(testset_x), columns=testset_x.columns)
    training_x = pd.DataFrame(scaler.fit_transform(training_x), columns=training_x.columns)

    print("training_x:", training_x.shape)  # vec/data
    print("training_y:", len(training_y))  # label
    print("testset_x:", testset_x.shape)  # vec/data
    print("testset_y:", len(testset_y))  # label

    training_x.to_csv(path + "training_x.csv")
    pd.DataFrame(training_y).to_csv(path + "training_y.csv")
    testset_x.to_csv(path + "testset_x.csv")
    pd.DataFrame(testset_y).to_csv(path + "testset_y.csv")

    # Training
    clf.fit(training_x, training_y)
    # Testing
    y_pred = clf.predict(testset_x)
    # Output test report
    print(metrics.classification_report(testset_y, y_pred))
    print("accuracy:", metrics.accuracy_score(testset_y, y_pred))

    try:
        f1_score_pos = metrics.f1_score(testset_y, y_pred, average=None)[1]   # get the f1-score for the positive class only
        print(metrics.f1_score(testset_y, y_pred, average=None)[1])
        tn, fp, fn, tp = metrics.confusion_matrix(testset_y, y_pred).ravel()
        print("@@@ tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    except IndexError:
        # this error must have meant that f1_score(...) returned only 1 value (f1 of negative class)
        if metrics.f1_score(testset_y, y_pred, average=None)[0] != 1:
            raise Exception('odd')
        print(metrics.f1_score(testset_y, y_pred, average=None))
        f1_score_pos = 0
        tn = 0
        fp = 0
        fn = 0
        tp = 0

    pos_at = list(clf.classes_).index(1)

    prob = clf.predict_proba(testset_x)[:, pos_at]

    auc = metrics.roc_auc_score(testset_y, prob)

    sorted_label = []
    order = np.argsort(prob)[::-1][:]  # numpy.ndarray
    # pos_all = sum([1 for label_real in testset_y if label_real == "yes"])
    pos_all = sum([1 for label_real in testset_y if label_real == 1])
    num_all = sum([1 for label_real in testset_y])
    print("number of samples:", num_all)
    total_recall = []
    length = []
    for i in order:
        a = testset_y[i]  # real label
        sorted_label.append(a)
        # pos_get = sum([1 for label_real in sorted_label if label_real == "yes"])
        pos_get = sum([1 for label_real in sorted_label if label_real == 1])
        length.append(len(sorted_label) / num_all)
        total_recall.append(pos_get / pos_all)
        # print(pos_get, len(sorted_label))
# ######
    total_recall = total_recall[::10]
    rate = length[::10]
    # append(1) in case that list out of range
    total_recall.append(1)
    rate.append(1)

    if type(stop_at) is tuple:
        stop_at = stop_at[0]

    stop = 0
    for index in range(len(total_recall)):
        if total_recall[index] >= stop_at:
            stop = index
            break

    print("AUC", auc)
    # print("pos_get", pos_get)
    # print("total recall stop_at", total_recall[stop])
    return rate[stop], auc, f1_score_pos, tn, fp, fn, tp


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    clf1 = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    clf2 = RandomForestClassifier()
    clf3 = tree.DecisionTreeClassifier()
    clf4 = KNeighborsClassifier(n_neighbors=1)
    clf5 = KNeighborsClassifier(n_neighbors=3)
    clf6 = KNeighborsClassifier(n_neighbors=5)
    clf7 = KNeighborsClassifier(n_neighbors=10)

    clf_list = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
    stopats = [1]

    path = r'csv/'

    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    interesing_list = sys.argv[1:] if len(sys.argv) >= 2 else [None]
    for argv in interesing_list:
        for clf in clf_list:  # SVM, ...
            print("@@@ A - classifier:", clf)
            for stopat_id in stopats:
                print("@@@ B - threshold stop at:", stopat_id)

                AUC = []
                cost = []
                f1_pos_list = []

                repeated_times = 10
                for i in range(1, 1+repeated_times):
                    print("\n-------------------------------------------------------------------------------------------------------------------")
                    print("@@@ C - Repeat number:", i, flush=True)
                    rate, auc, f1_pos, tn, fp, fn, tp = main(path, stop_at=stopat_id,
                                        seed=42 + i, clf=clf, interesting_path=argv)
                    AUC.append(auc)
                    cost.append(rate)
                    f1_pos_list.append(f1_pos)

                    # total_tn += tn
                    # total_fp += fp
                    # total_fn += fn
                    # total_tp += tp

                f1_pos_med = statistics.median(f1_pos_list)
                AUC_med = statistics.median(AUC)
                AUC_iqr = np.subtract(*np.percentile(AUC, [75, 25]))
                COST_med = statistics.median(cost)
                COST_iqr = np.subtract(*np.percentile(cost, [75, 25]))
                print("-----------------------------FINAL RESULT---------------------------------")
                print("----------threshold stop at----------:", stopat_id)
                print('AUC', AUC)
                print("AUC_median", AUC_med)
                print("AUC_iqr", AUC_iqr)
                print('cost', cost)
                print("COST_med", COST_med)
                print("COST_iqr", COST_iqr)
                print("f1_pos_list", f1_pos_list)
                print("f1_pos_med", f1_pos_med)

    # print('total_tn', total_tn)
    # print('total_fp', total_fp)
    # print('total_fn', total_fn)
    # print('total_tp', total_tp)

    # final_p = total_tp / (total_tp + total_fp)
    # print('p=', final_p)
    # final_r = total_tp / (total_tp + total_fn)
    # print('r=', final_r)
    # print('f=', 2 * final_p * final_r / (final_p + final_r))

