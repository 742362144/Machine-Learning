'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

# TODO 使用说明
# TODO 1 使用随机梯度下降训练权值
# TODO 2 根据权值和输入样本进行分类


class LogisiticRegression(object):

    def __init__(self):
        print("hello LogisiticRegression!")

    # sigmoid函数
    # inX---实数
    def sigmoid(self, inX):
        return 1.0/(1+exp(-inX))

    def gradAscent(self, dataMatIn, classLabels):
        dataMatrix = mat(dataMatIn)             # 转化为矩阵
        labelMat = mat(classLabels).transpose()  # 转化为矩阵并求转置
        m, n = shape(dataMatrix)  # 获取训练数据集的维数
        alpha = 0.001  # 学习率(梯度上升时的步数大小，太大会造成震荡不收敛，太小收敛速度慢，总有一个足够小的学习率使得其收敛)
        maxCycles = 500  # 迭代次数
        weights = ones((n, 1))  # 权值初始化为1
        for k in range(maxCycles):  # heavy on matrix operations
            h = self.sigmoid(dataMatrix*weights)  # 利用sigmoid来求得概率
            error = (labelMat - h)  # 计算误差
            weights = weights + alpha * dataMatrix.transpose() * error  # 梯度上升
        return weights

    # 随机梯度上升(改进版)
    # 在线学习算法
    # 一次仅使用一个样本来对分类器进行增量式更新
    def stocGradAscent(self, dataMatrix, classLabels, numIter=150):
        m, n = shape(dataMatrix)
        weights = ones(n)   #initialize to all ones
        for j in range(numIter):  # 默认迭代150次
            dataIndex = range(m)
            for i in range(m):
                alpha = 4/(1.0+j+i)+0.0001    # 学习率在每次迭代时更新，随迭代次数减少，但不会减少到0
                randIndex = int(random.uniform(0, len(dataIndex)))  # 通过随机选取样本来更新回归系数
                h = self.sigmoid(sum(dataMatrix[randIndex]*weights))  # 每次随机算取一个样本来对分类器进行增量式更新
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]  # 一次仅使用一个样本点来更新回归系数
                del(dataIndex[randIndex])  # 删除使用过的样本下标(相当于无放回的选取)
        return weights

    def classifyVector(self, inX, weights):
        prob = self.sigmoid(sum(inX * weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0
