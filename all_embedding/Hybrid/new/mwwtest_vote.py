import scipy.stats as st
import csv
import os

def read_csv_file(path):
    with open(path, mode='rt', encoding='UTF-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        # col = [float(row["f1"].replace("%", "")) for row in reader]
        col = [round(float(row['f1']),15) for row in reader]
    return col


def read_file(path):
    f1_score = 0
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(1, len(lines)):
        f1_score += float(lines[i].split("\t")[4])
    f1_score = f1_score/(len(lines) - 1)
    return f1_score

if __name__ == '__main__':
    path_base = "/mnt/7T/mingwen1/data/bugDetection/srcIRcom/all_embedding/app/vote_score/"
    results = []
    for i in range(2, 11):
        vote_score_path = path_base+"vote_group"+str(i)+"/soft_vote_results"
        results.append((i, read_file(vote_score_path)))
    results = sorted(results, key=lambda k: k[1], reverse=True)
    print("soft", results)
    results = []
    for i in range(2, 11):
        vote_score_path = path_base+"vote_group"+str(i)+"/hard_vote_results"
        results.append((i, read_file(vote_score_path)))
    results = sorted(results, key=lambda k: k[1], reverse=True)
    print("hard", results)

    # path_vote = "/mnt/7T/mingwen1/data/bugDetection/srcIRcom/all_embedding/app/vote_score/"
    # path_single_base = "/mnt/7T/mingwen1/data/bugDetection/srcIRcom/all_embedding/app/vote_data/"
    # for index in ["single_group"]:
    #     vote_result_path = path_vote+"vote_"+index+"/soft_vote_results.csv"
    #     vote_f1 = read_csv_file(vote_result_path)
    #     for model in os.listdir(path_single_base):
    #         model_result_path = path_single_base+model+"/best_score_record.csv"
    #         single_f1 = read_csv_file(model_result_path)
    #         result = st.mannwhitneyu(vote_f1, single_f1, alternative='greater')
    #         p_value = result.pvalue
    #         if p_value > 0.05:
    #             print(model+" do not greater than "+index)
    #             print("p_value", p_value)