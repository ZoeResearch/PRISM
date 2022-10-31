import collections
import pickle
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

if __name__ == '__main__':
    dic_statistic = {}
    dic_doc = {}
    violation_doc_path = "../app/data/src_top_20_with_time_newest_with_warning_context_for_warning_type_without_test_classes"
    all_code = pickle.load(open(violation_doc_path, "rb"))
    print()
    for codedoc in all_code:
        if codedoc.cls == 1:
            pattern = codedoc.info["violation_type"]
            if pattern not in dic_statistic.keys():
                dic_statistic[pattern] = 1
                dic_doc[pattern] = []
                dic_doc[pattern].append(codedoc)
            else:
                dic_statistic[pattern] += 1
                dic_doc[pattern].append(codedoc)
    dic_statistic = sorted(dic_statistic.items(), key = lambda kv:(kv[1], kv[0]))
    print()

