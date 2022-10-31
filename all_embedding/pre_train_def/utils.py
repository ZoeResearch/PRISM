import re
import pickle
import collections
import os
import javalang
import gc

CodeDocument = collections.namedtuple('CodeDocument', 'words tags cls info')
# 3 6 9 12 毕业
def read_file(doc_path):
    # f = open(doc_path, "rb")
    all_code_blocks = []
    all_labels = []
    num = 0
    num1 = 0
    null_num = 0
    # print(len(pickle.load(f)))
    with open(doc_path, "rb") as f:
        blocks = pickle.load(f)
        for block in blocks:
            if block.words:
                try:
                    code = block.words
                    all_code_blocks.append(code)
                    all_labels.append(block.cls)
                except UnicodeEncodeError as e:
                    print(block)
                    print(e)
                    num += 1

            else:
                null_num += 1
    print("unicode error:", num)
    print("lex error:", num1)
    print("null:", null_num)

    return all_code_blocks, all_labels



def read_ir_file(doc_path):
    f = open(doc_path, "rb")
    all_code_blocks = []
    all_labels = []
    num = 0
    num1 = 0
    null_num = 0
    # print(len(pickle.load(f)))
    for block in pickle.load(f):
        if block.words:
            try:
                " ".join(block.words).encode(encoding='UTF-8',errors='strict')
                code = []
                j = 0
                while j+1<len(block.words):
                    if block.words[j] == ":" and block.words[j+1] == "=":
                        code.append(":=")
                        j += 2
                    else:
                        code.append(block.words[j])
                        j += 1
                # for i in range(len(code)):
                #     if re.match("\".*\"", code[i].strip()):
                #         code[i] = re.sub("\".*\"", "CONST", code[i].strip())
                #     if re.match("\'.*\'", code[i].strip()):
                #         code[i] = re.sub("\'.*\'", "CONST", code[i].strip())
                all_code_blocks.append(code)
                all_labels.append(block.cls)
            except UnicodeEncodeError as e:
                print(block)
                print(e)
                num += 1
            except javalang.tokenizer.LexerError as e:
                num1 += 1
        else:
            null_num += 1
    print("unicode error:", num)
    print("lex error:", num1)
    print("null:", null_num)

    return all_code_blocks, all_labels
