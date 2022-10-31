import numpy as np
from numpy import *

class PCA():
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def pca(self):
        data = np.asarray(self.dataset).reshape(len(self.dataset), -1)
        meanVals = mean(data, axis=0)
        DataAdjust = data - meanVals           #减去平均值
        covMat = cov(DataAdjust, rowvar=0)
        eigVals,eigVects = linalg.eig(mat(covMat)) #计算特征值和特征向量
        #print eigVals
        eigValInd = argsort(eigVals)
        eigValInd = eigValInd[:-(self.k+1):-1]   #保留最大的前K个特征值
        redEigVects = eigVects[:,eigValInd]        #对应的特征向量
        lowDDataMat = DataAdjust * redEigVects     #将数据转换到低维新空间
        # reconMat = (lowDDataMat * redEigVects.T) + meanVals   #重构数据，用于调试
        return lowDDataMat
