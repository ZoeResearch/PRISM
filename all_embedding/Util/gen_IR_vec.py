import sys
sys.path.append("all_embedding/")
# from Util.utils import *
import javalang
import re
import numpy as np

def get_IR_embedding(model, code_blocks, voc_size, sentence_length, code, opcodeIDF):
    # code_blocks中每一项都是一个 行信息组成的block列表
    # embedding of each token
    blockTokenToVector = []
    # used for calculate tf
    blockToOpcodeNum = []
    blockToOpcodeCounts = []
    opcodeTF = []
    # used for calculate idf
    # opcodeToBlockCounts = {}
    # opcodeIDF = {}
#构建一个lineObject  line需要有序
    for code_snip in code_blocks:
        tokenToVector = {}
        opcode_num = 0
        opcode_count = {}
        for line in code_snip.words:
        # for line in code_snip:
            try:
                for key, values in line["operator"].items():
                    for value in values:
                        tokenToVector[value] = list(model[value])
                        opcode_num += 1
                        if value not in opcode_count.keys():
                            opcode_count[value] = 1
                        else:
                            opcode_count[value] += 1

                for key, values in line["operand"].items():
                    for value in values:
                        tokenToVector[value] = list(model[value])
            except KeyError as e:
                print(e)
                print(code_snip)
                # exit(0)
            except TypeError as e:
                print(e)
                print(code_snip)
                # exit(0)
        blockTokenToVector.append(tokenToVector)  # 每个block中含有该block中所有指令的向量值
        blockToOpcodeNum.append(opcode_num)  # 每个值是该指令所含的opcode数量   注意可能有全0除数出现
        blockToOpcodeCounts.append(opcode_count)  # 每个block是该block中所有指令的频次

    if len(blockToOpcodeNum) != len(blockToOpcodeCounts):
        print("get problem")
        exit(0)

    for i in range(len(blockToOpcodeNum)):
        single_opcode_tf = {}
        for opcode in blockToOpcodeCounts[i].keys():
            single_opcode_tf[opcode] = blockToOpcodeCounts[i][opcode] / blockToOpcodeNum[i]
            # if opcode not in opcodeToBlockCounts.keys():
            #     opcodeToBlockCounts[opcode] = 0
        opcodeTF.append(single_opcode_tf)  # 每一项是一个block，有该block里有所含指令的tf值

    # for opcode_count in blockToOpcodeCounts:
    #     for opcode in opcode_count.keys():
    #         opcodeToBlockCounts[opcode] += 1  # 每一项是出现该opcode的文档个数

    # for opcode in opcodeToBlockCounts.keys():
    #     opcodeIDF[opcode] = np.log10(len(code_blocks) / opcodeToBlockCounts[opcode])  # 每一项是该opcode的idf值

    feature_vectors = concatenate_embedding(code_blocks, blockTokenToVector, opcodeTF, opcodeIDF, voc_size, code)
    return feature_vectors

def concatenate_embedding(code_blocks, blockTokenToVector, opcodeTF, opcodeIDF, voc_size, code):
    feature_vectors = []
    for k in range(len(code_blocks)):
        if code_blocks[k].words:
            opcode_tf = opcodeTF[k]
            token2vec = blockTokenToVector[k]
            single_line_feature_vector = []
            for line in code_blocks[k].words:
            # for line in code_blocks[k]:
                for key in line["operator"].keys():
                    opcode_vector = np.zeros(voc_size)
                    operand_vector = np.zeros(voc_size)
                    try:
                        for token in line["operator"][key]:
                            if code == "ir_id_1":
                                if token == "=":
                                    continue
                                opcode_vector += np.asarray(token2vec[token]) * opcode_tf[token] * opcodeIDF[token] * 100  # note:change added number
                            elif code == "ir_id_2":
                                opcode_vector += np.asarray(token2vec[token])
                        for values in line["operand"][key]:
                            operand_vector += np.asarray(token2vec[values])
                    except KeyError as e:
                        print(e)
                        # exit(0)
                    single_line_feature_vector.append(opcode_vector)
                    single_line_feature_vector.append(operand_vector)
            feature_vectors.append(single_line_feature_vector)
    return feature_vectors


























