# @author: Alan
# @Date: 2021/4/7
# @Time: 上午10:52
# @Email: 2785941910@qq.com
# @File: word2vec_lstm.py

import sys
sys.path.append("../")
import numpy as np

allIns = ['nop', 'aconst_null', 'iconst_m1', 'iconst_0', 'iconst_1', 'iconst_2', 'iconst_3', 'iconst_4',
                 'iconst_5', 'lconst_0', 'lconst_1', 'fconst_0', 'fconst_1', 'fconst_2', 'dconst_0', 'dconst_1',
                 'bipush', 'sipush', 'ldc', 'ldc_w', 'ldc2_w', 'iload', 'lload', 'fload', 'dload', 'aload', 'iload_0',
                 'iload_1', 'iload_2', 'iload_3', 'lload_0', 'lload_1', 'lload_2', 'lload_3', 'fload_0', 'fload_1',
                 'fload_2', 'fload_3', 'dload_0', 'dload_1', 'dload_2', 'dload_3', 'aload_0', 'aload_1', 'aload_2',
                 'aload_3', 'iaload', 'laload', 'faload', 'daload', 'aaload', 'baload', 'caload', 'saload', 'istore',
                 'lstore', 'fstore', 'dstore', 'astore', 'istore_0', 'istore_1', 'istore_2', 'istore_3', 'lstore_0',
                 'lstore_1', 'lstore_2', 'lstore_3', 'fstore_0', 'fstore_1', 'fstore_2', 'fstore_3', 'dstore_0',
                 'dstore_1', 'dstore_2', 'dstore_3', 'astore_0', 'astore_1', 'astore_2', 'astore_3', 'iastore',
                 'lastore', 'fastore', 'dastore', 'aastore', 'bastore', 'castore', 'sastore', 'pop', 'pop2', 'dup',
                 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2', 'swap', 'iadd', 'ladd', 'fadd', 'dadd', 'isub',
                 'lsub', 'fsub', 'dsub', 'imul', 'lmul', 'fmul', 'dmul', 'idiv', 'ldiv', 'fdiv', 'ddiv', 'irem',
                 'lrem', 'frem', 'drem', 'ineg', 'lneg', 'fneg', 'dneg', 'ishl', 'lshl', 'ishr', 'lshr', 'iushr',
                 'lushr', 'iand', 'land', 'ior', 'lor', 'ixor', 'lxor', 'iinc', 'i2l', 'i2f', 'i2d', 'l2i', 'l2f',
                 'l2d', 'f2i', 'f2l', 'f2d', 'd2i', 'd2l', 'd2f', 'i2b', 'i2c', 'i2s', 'lcmp', 'fcmpl', 'fcmpg',
                 'dcmpl', 'dcmpg', 'ifeq', 'ifne', 'iflt', 'ifge', 'ifgt', 'ifle', 'if_icmpeq', 'if_icmpne',
                 'if_icmplt', 'if_icmpge', 'if_icmpgt', 'if_icmple', 'if_acmpeq', 'if_acmpne', 'goto', 'jsr', 'ret',
                 'tableswitch', 'lookupswitch', 'ireturn', 'lreturn', 'freturn', 'dreturn', 'areturn', 'return',
                 'getstatic', 'putstatic', 'getfield', 'putfield', 'invokevirtual', 'invokespecial', 'invokestatic',
                 'invokeinterface', 'invokedynamic', 'new', 'newarray', 'anewarray', 'arraylength', 'athrow',
                 'checkcast', 'instanceof', 'monitorenter', 'monitorexit', 'wide', 'multianewarray', 'ifnull',
                 'ifnonnull', 'goto_w', 'jsr_w', 'breakpoint', 'impdep1', 'impdep2']

def regulate(allIns):
    ignore_list = ['if_icmpeq', 'if_icmpne', 'if_icmplt', 'if_icmpge', 'if_icmpgt', 'if_icmple', 'if_acmpeq',
                   'if_acmpne', 'goto_w', 'jsr_w']
    regulatedIns = set()
    for i in range(len(allIns)):
        if "_" in allIns[i] and allIns[i] not in ignore_list:
            regulatedIns.add(allIns[i].split("_")[0])
        else:
            regulatedIns.add(allIns[i])
    return regulatedIns

def get_byte_embedding(model, code_blocks, voc_size, opcodeIDF, code):
    # embedding of each token
    blockTokenToVector = []
    # used for calculate tf
    blockToOpcodeNum = []
    blockToOpcodeCounts = []
    opcodeTF = []
    # used for calculate idf
    opcodeToBlockCounts = {}
    # opcodeIDF = {}
    num = 0
    ins = regulate(allIns)
    for code_snip in code_blocks:
        tokenToVector = {}
        opcode_num = 0
        opcode_count = {}
        try:
            for token in code_snip:
                tokenToVector[token] = list(model[token])
                if token in ins:
                    opcode_num += 1
                    if token not in opcode_count.keys():
                        opcode_count[token] = 1
                    else:
                        opcode_count[token] += 1
        except KeyError as e:
            print(code_snip)
            print(e)
            num += 1
        blockTokenToVector.append(tokenToVector)              # 每个block中含有该block中所有指令的向量值
        blockToOpcodeNum.append(opcode_num)                   # 每个值是该指令所含的opcode数量   注意可能有全0除数出现
        blockToOpcodeCounts.append(opcode_count)              # 每个block是该block中所有指令的频次

    if len(blockToOpcodeNum) != len(blockToOpcodeCounts):
        print("get problem")

    for i in range(len(blockToOpcodeNum)):
        single_opcode_tf = {}
        for opcode in blockToOpcodeCounts[i].keys():
            single_opcode_tf[opcode] = blockToOpcodeCounts[i][opcode] / blockToOpcodeNum[i]
            # if opcode not in opcodeToBlockCounts.keys():
            #     opcodeToBlockCounts[opcode] = 0
        opcodeTF.append(single_opcode_tf)       #每一项是一个block，有该block里有所含指令的tf值

    # for opcode_count in blockToOpcodeCounts:
    #     for opcode in opcode_count.keys():
    #         opcodeToBlockCounts[opcode] += 1    # 每一项是出现该opcode的文档个数

    # for opcode in opcodeToBlockCounts.keys():
    #     opcodeIDF[opcode] = np.log10(len(code_blocks)/opcodeToBlockCounts[opcode])       # 每一项是该opcode的idf值
    if code == "byte_id_1":
        feature_vectors = concatenate_embedding(code_blocks, model, opcodeTF, opcodeIDF, voc_size, allIns)
    elif code == "byte_id_2":
        feature_vectors = concatenate_embedding(code_blocks, model, opcodeTF, opcodeIDF, voc_size, ins)
    elif code == "byte_id_3":
       feature_vectors = concatenate_embedding_ignore(code_blocks, model, opcodeTF, opcodeIDF, voc_size, ins)

    return feature_vectors

def concatenate_embedding(code_blocks, model, opcodeTF, opcodeIDF, voc_size, ins):
    feature_vectors = []
    for k in range(len(code_blocks)):
        if code_blocks[k]:
            single_feature_vector = []
            opcode_tf = opcodeTF[k]
            line_length = len(code_blocks[k])
            i = 0
            num = 0
            while i < line_length:
                try:
                    if code_blocks[k][i] in ins:
                        #model[]? is array?
                        opcode_vector = model[code_blocks[k][i]] * opcode_tf[code_blocks[k][i]] * opcodeIDF[code_blocks[k][i]]
                        single_feature_vector.append(opcode_vector)
                        operand_vector = np.zeros(voc_size)
                        if i + 1 == line_length:
                            single_feature_vector.append(opcode_vector)
                            break
                        else:
                            for j in range(i + 1, line_length):
                                if code_blocks[k][j] in ins:
                                    if sum(operand_vector) != 0:
                                        single_feature_vector.append(operand_vector)
                                        i = j
                                        break
                                    else:
                                        i = j
                                        break
                                else:
                                    operand_vector += model[code_blocks[k][j]]
                                    if j + 1 == line_length:
                                        single_feature_vector.append(operand_vector)
                                        i = line_length
                                        break
                    else:
                        i += 1
                except KeyError as e:
                    # print(e)
                    num += 1
                    i += 1

            feature_vectors.append(single_feature_vector)
    print("KeyError:", num)
    return feature_vectors

def concatenate_embedding_ignore(code_blocks, model, opcodeTF, opcodeIDF, voc_size, ins):
    ignoreList = ["Field", "Method", "Class", "CONST"]
    feature_vectors = []
    for k in range(len(code_blocks)):
        if code_blocks[k]:
            single_feature_vector = []
            opcode_tf = opcodeTF[k]
            line_length = len(code_blocks[k])
            i = 0
            num = 0
            while i < line_length:
                try:
                    if code_blocks[k][i] in ins:
                        #model[]? is array?
                        opcode_vector = model[code_blocks[k][i]] * opcode_tf[code_blocks[k][i]] * opcodeIDF[code_blocks[k][i]]
                        single_feature_vector.append(opcode_vector)
                        operand_vector = np.zeros(voc_size)
                        if i + 1 == line_length:
                            single_feature_vector.append(opcode_vector)    #监控opcode在末尾
                            break
                        else:
                            for j in range(i + 1, line_length):
                                if code_blocks[k][j] in ins:
                                    if sum(operand_vector) != 0:
                                        single_feature_vector.append(operand_vector)
                                        i = j
                                        break
                                    else:
                                        i = j
                                        break
                                else:
                                    if code_blocks[k][j] in ignoreList:
                                        while(code_blocks[k][j] not in ins):
                                            # print("ignore tokens:", code_blocks[k][j])
                                            if j+1==line_length:
                                                break
                                            j += 1

                                        i = j
                                        break
                                    operand_vector += model[code_blocks[k][j]]
                                    if j + 1 == line_length:             #监控operand在末尾
                                        single_feature_vector.append(operand_vector)
                                        i = line_length
                                        break
                    else:
                        i += 1
                except KeyError as e:
                    # print(e)
                    num += 1
                    i += 1

            feature_vectors.append(single_feature_vector)
    print("KeyError:", num)
    return feature_vectors

def concatenate_embedding_by_add(code_blocks, model, opcodeTF, opcodeIDF, voc_size, sentence_length):
    feature_vectors = []
    for k in range(len(code_blocks)):
        # print("k---", k)  #
        single_feature_vector = np.zeros(sentence_length, voc_size*2)
        opcode_tf = opcodeTF[k]
        line_length = len(code_blocks[k])
        i = 0
        while i < line_length:
            if code_blocks[k][i] in allIns:
                line_code = []
                line_code.append(code_blocks[k][i])
                weight = opcode_tf[code_blocks[k][i]] * opcodeIDF[code_blocks[k][i]]
                # weight = 1
                line_vector = model[code_blocks[k][i]] * weight
                operand_vectors = np.zeros(voc_size)
                if i+1 == line_length:
                    single_feature_vector += np.hstack((line_vector, operand_vectors))
                    break
                else:
                    for j in range(i+1, line_length):
                        if code_blocks[k][j] in allIns:
                            line_vector = np.hstack((line_vector, operand_vectors))
                            single_feature_vector += line_vector
                            i = j
                            break
                        else:
                            operand_vectors += model[code_blocks[k][j]]
                            if j+1 == line_length:
                                line_vector = np.hstack((line_vector, operand_vectors))
                                single_feature_vector += line_vector
                                i = line_length
                                break
        feature_vectors.append([single_feature_vector])

    return feature_vectors

