import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def linear_row(all_src_vec, all_ir_vec, all_byte_vec):
    splice_all = []
    for i in range(len(all_src_vec)):
        splice = list(all_src_vec[i]) + list(all_ir_vec[i]) + list(all_byte_vec[i])
        splice_all.append(splice)
    return np.array(splice_all)

def linear_col(all_src_vec, all_ir_vec, all_byte_vec):
    splice_all = []
    for i in range(len(all_src_vec)):
        splice = np.hstack((all_src_vec[i], all_ir_vec[i], all_byte_vec[i]))
        splice_all.append(splice)
    return np.array(splice_all)

def linear_highDim(all_src_vec, all_ir_vec, all_byte_vec):   #不知三维如何输入lstm?
    splice_all = []
    for i in range(len(all_src_vec)):
        splice = []
        splice.append(all_src_vec[i])
        splice.append(all_ir_vec[i])
        splice.append(all_byte_vec[i])
        splice_all.append(np.array(splice))
    return np.array(splice_all)

def add(all_src_vec, all_ir_vec, all_byte_vec):
    return np.array(all_src_vec) + np.array(all_ir_vec) + np.array(all_byte_vec)

def sub(all_src_vec, all_ir_vec, all_byte_vec):
    return np.array(all_src_vec) - np.array(all_ir_vec) - np.array(all_byte_vec)

def Hadamard_product(all_src_vec, all_ir_vec, all_byte_vec):
    return np.array(all_src_vec) * np.array(all_ir_vec) * np.array(all_byte_vec)

def Cosine_similarity(all_src_vec, all_ir_vec, all_byte_vec):
    all_similarity = []
    all_similarity_mean = []
    for i in range(len(all_src_vec)):
        similarity = np.vstack((cosine_similarity(all_src_vec[i], all_ir_vec[i])\
                          , cosine_similarity(all_src_vec[i], all_byte_vec[i])\
                          , cosine_similarity(all_byte_vec[i], all_ir_vec[i]))   )
        all_similarity.append(similarity)
        all_similarity_mean.append(np.mean(similarity, axis=1))
    return np.array(all_similarity), np.array(all_similarity_mean)

def Eucl_distance(A, B):
    A = np.array(A)
    B = np.array(B)
    return np.array(np.sqrt(-2 * np.dot(A, B.T) + np.sum(np.square(B), axis=1) + np.transpose(
        [np.sum(np.square(A), axis=1)])))

def distance(all_src_vec, all_ir_vec, all_byte_vec):
    all_distance = []
    for i in range(len(all_src_vec)):
        distance = np.vstack((Eucl_distance(all_src_vec[i], all_ir_vec[i])\
                          , Eucl_distance(all_src_vec[i], all_byte_vec[i])\
                          , Eucl_distance(all_byte_vec[i], all_ir_vec[i])))
        all_distance.append(distance)
    return np.array(all_distance)

def splice(splice_all_lin, splice_all_add, splice_all_sub, splice_all_had):
    all = []
    for i in range(len(splice_all_lin)):
        # all_matrix = np.vstack((np.hstack((np.vstack((splice_all_lin[i], splice_all_add[i], splice_all_sub[i], splice_all_had[i])), \
        #                                    np.zeros((sentence_len*6, sentence_len - vec_len)))), splice_all_euc[i])

        all_matrix = np.vstack((splice_all_lin[i], splice_all_add[i], splice_all_sub[i], splice_all_had[i]))
        all.append(all_matrix)
    return np.array(all)