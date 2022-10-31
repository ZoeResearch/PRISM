import pickle
import collections
CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')

if __name__ == '__main__':
    violation_doc_path = "./data/src_top_35_without_deduplication_with_constructor_with_baseline"
    save_path = "./data/src_top_35_without_deduplication_with_constructor_with_baseline_dedup_by_class"
    # violation_doc_path = "./data/src_top_20_with_time_newest_with_warning_context_for_warning_type_without_test_classes"
    # violation_doc_path = "./data/src_top_20"
    all_code = pickle.load(open(violation_doc_path, "rb"))

    utils = []
    dedup_result = []
    fix_num = 0
    unfix_num = 0

    for item in all_code:
        if item.info["type"] == "fixed":
            if item.words not in utils:
                dedup_result.append(item)
                utils.append(item.words)
                fix_num += 1
        else:
            unfix_num += 1

    print("fixed:", fix_num)
    # print("unfixed:", unfix_num)
    # for item in all_code:
    #     if item.words not in utils:
    #         dedup_result.append(item)
    #         utils.append(item.words)
    #         if item.info["type"] == "fixed":
    #             fix_num += 1
    #     else:
    #         continue
    # print("all:", len(dedup_result))
    # print("unfixed:", len(dedup_result)-fix_num)
    # print("fixed:", fix_num)