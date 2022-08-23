import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def linear_vstack(all_src_vec, all_ir_vec, all_byte_vec):
#     splice_all = []
#     for i in range(len(all_src_vec)):
#         # splice = list(all_src_vec[i]) + list(all_ir_vec[i]) + list(all_byte_vec[i])
#         splice = single_data_vstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
#         splice_all.append(splice)
#     return np.asarray(splice_all).astype(np.float32)

# def linear_hstack(all_src_vec, all_ir_vec, all_byte_vec):
#     splice_all = []
#     for i in range(len(all_src_vec)):
#         # splice = np.hstack((all_src_vec[i], all_ir_vec[i], all_byte_vec[i]))
#         splice = single_data_hstack(all_src_vec[i], all_ir_vec[i], all_byte_vec[i])
#         splice_all.append(splice)
#     return np.array(splice_all)

def linear_hstack(*args):
    splice_all = []
    for i in range(len(args[0])):
        single = []
        for j in range(len(args)):
            single.append(args[j][i])
        splice = single_data_hstack(single)
        splice_all.append(splice)
    return np.array(splice_all)

def linear_vstack(*args):
    splice_all = []
    for i in range(len(args[0])):
        single = []
        for j in range(len(args)):
            single.append(args[j][i])
        splice = single_data_vstack(single)
        splice_all.append(splice)
    return np.array(splice_all)

def single_data_hstack(single):
    all = single[0]
    for i in range(1, len(single)):
        all = np.hstack((all, single[i]))
    return all

def single_data_vstack(single):
    all = single[0]
    for i in range(1, len(single)):
        all = np.vstack((all, single[i]))
    return all

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos
#预余弦距离用1-余弦相似度
def one_dim_cos_sim(src_flatten, ir_flatten, byte_flattern):
    sim1 = cos_sim(src_flatten, ir_flatten)
    sim2 = cos_sim(src_flatten, byte_flattern)
    sim3 = cos_sim(ir_flatten, byte_flattern)
    return np.array([sim1, sim2, sim3])

def flatten_hstack(all_src_vec, all_ir_vec, all_byte_vec, sentence_length_range, voc_size):
    splice_all = []
    for i in range(len(all_src_vec)):
        src_flatten = np.array(all_src_vec[i]).reshape(1, sentence_length_range[0] * voc_size)
        ir_flatten = np.array(all_ir_vec[i]).reshape(1, sentence_length_range[1] * voc_size)
        byte_flattern = np.array(all_byte_vec[i]).reshape(1, sentence_length_range[2] * voc_size)
        sim = one_dim_cos_sim(src_flatten[0], ir_flatten[0], byte_flattern[0]).reshape(1, 3)
        splice = linear_hstack(src_flatten, ir_flatten, byte_flattern, sim)
        splice_all.append(splice)
    return np.asarray(splice_all)

def linear_highDim(all_src_vec, all_ir_vec, all_byte_vec):   #不知三维如何输入lstm?
    splice_all = []
    for i in range(len(all_src_vec)):
        splice = []
        splice.append(all_src_vec[i])
        splice.append(all_ir_vec[i])
        splice.append(all_byte_vec[i])
        splice_all.append(np.array(splice))
    return np.array(splice_all)

def add(all_src_vec, all_ir_vec, all_byte_vec, max_length, voc_size):
    all_vec = []
    for i in range(len(all_src_vec)):
        vecs = []
        for vec in [all_src_vec[i], all_ir_vec[i], all_byte_vec[i]]:
            new_vec = []
            for i in range(len(vec)):
                new_vec.append(vec[i])
            for j in range(len(vec), max_length):
                new_vec.append([0] * voc_size)
            vecs.append(np.asarray(new_vec).astype(np.float32))
        all_vec.append(vecs[0] + vecs[1] + vecs[2])

    return np.asarray(all_vec).astype(np.float32)

def sub(all_src_vec, all_ir_vec, all_byte_vec,max_length, voc_size):
    all_vec = []
    for i in range(len(all_src_vec)):
        vecs = []
        for vec in [all_src_vec[i], all_ir_vec[i], all_byte_vec[i]]:
            new_vec = []
            for i in range(len(vec)):
                new_vec.append(vec[i])
            for j in range(len(vec), max_length):
                new_vec.append([0]*voc_size)
            vecs.append(np.asarray(new_vec).astype(np.float32))
        all_vec.append(vecs[0]-vecs[1]-vecs[2])

    return np.asarray(all_vec).astype(np.float32)

# def sub(all_src_vec, all_byte_vec,max_length, voc_size):
#     all_vec = []
#     for i in range(len(all_src_vec)):
#         vecs = []
#         for vec in [all_src_vec[i], all_byte_vec[i]]:
#             new_vec = []
#             for i in range(len(vec)):
#                 new_vec.append(vec[i])
#             for j in range(len(vec), max_length):
#                 new_vec.append([0]*voc_size)
#             vecs.append(np.asarray(new_vec).astype(np.float32))
#         all_vec.append(vecs[0]-vecs[1]-vecs[2])
#
#     return np.asarray(all_vec).astype(np.float32)

def mul(all_src_vec, all_ir_vec, all_byte_vec,max_length, voc_size):
    all_vec = []
    for i in range(len(all_src_vec)):
        vecs = []
        for vec in [all_src_vec[i], all_ir_vec[i], all_byte_vec[i]]:
            new_vec = []
            for i in range(len(vec)):
                new_vec.append(vec[i])
            for j in range(len(vec), max_length):
                new_vec.append([1] * voc_size)
            vecs.append(np.asarray(new_vec).astype(np.float32))
        all_vec.append(vecs[0] * vecs[1] * vecs[2])
    return np.asarray(all_vec).astype(np.float32)

def Cosine_similarity_hstack(all_src_vec, all_ir_vec, all_byte_vec):
    all_similarity = []
    all_similarity_mean = []
    for i in range(len(all_src_vec)):
        similarity = np.hstack((cosine_similarity(all_src_vec[i], all_ir_vec[i])
                          , cosine_similarity(all_src_vec[i], all_byte_vec[i])
                          , cosine_similarity(all_byte_vec[i], all_ir_vec[i])))
        all_similarity.append(similarity)
        all_similarity_mean.append(np.mean(similarity, axis=1))
    return np.array(all_similarity)

def Cosine_similarity_vstack(all_src_vec, all_ir_vec, all_byte_vec):
    all_similarity = []
    all_similarity_mean = []
    for i in range(len(all_src_vec)):
        similarity = np.vstack((cosine_similarity(all_src_vec[i], all_ir_vec[i])
                          , cosine_similarity(all_src_vec[i], all_byte_vec[i])
                          , cosine_similarity(all_byte_vec[i], all_ir_vec[i])))
        all_similarity.append(similarity)
        all_similarity_mean.append(np.mean(similarity, axis=1))
    return np.array(all_similarity)

def Eucl_distance(A, B):
    A = np.array(A)
    B = np.array(B)
    return np.array(np.sqrt(-2 * np.dot(A, B.T) + np.sum(np.square(B), axis=1) + np.transpose(
        [np.sum(np.square(A), axis=1)])))

def Eucl_distance_hstack(all_src_vec, all_ir_vec, all_byte_vec):
    all_distance = []
    for i in range(len(all_src_vec)):
        distance = np.hstack((Eucl_distance(all_src_vec[i], all_ir_vec[i])\
                          , Eucl_distance(all_src_vec[i], all_byte_vec[i])\
                          , Eucl_distance(all_byte_vec[i], all_ir_vec[i])))
        all_distance.append(distance)
    return np.asarray(all_distance).astype(np.float32)

def Eucl_distance_vstack(all_src_vec, all_ir_vec, all_byte_vec):
    all_distance = []
    for i in range(len(all_src_vec)):
        distance = np.vstack((Eucl_distance(all_src_vec[i], all_ir_vec[i])\
                          , Eucl_distance(all_src_vec[i], all_byte_vec[i])\
                          , Eucl_distance(all_byte_vec[i], all_ir_vec[i])))
        all_distance.append(distance)
    return np.asarray(all_distance).astype(np.float32)

def splice_hstack(splice_all_lin, splice_all_add, splice_all_sub, splice_all_had):
    all = []
    for i in range(len(splice_all_lin)):
        all_matrix = np.hstack((splice_all_lin[i], splice_all_add[i], splice_all_sub[i], splice_all_had[i]))
        all.append(all_matrix)
    return np.asarray(all).astype(np.float32)

def splice_vstack(splice_all_lin, splice_all_add, splice_all_sub, splice_all_had):
    all = []
    for i in range(len(splice_all_lin)):
        # all_matrix = np.vstack((np.hstack((np.vstack((splice_all_lin[i], splice_all_add[i], splice_all_sub[i], splice_all_had[i])), \
        #                                    np.zeros((sentence_len*6, sentence_len - vec_len)))), splice_all_euc[i])

        all_matrix = np.vstack((splice_all_lin[i], splice_all_add[i], splice_all_sub[i], splice_all_had[i]))
        all.append(all_matrix)
    return np.asarray(all).astype(np.float32)

# def splice_vstack(part_data, max_length, voc_size):
#     all = []
#     length = len(part_data[0])
#     for i in range(length):
#         for j in range(len(part_data)):
