'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

# TODO 使用说明
# TODO 1 使用trainNB0得出类别1和类别0时的条件概率
# TODO 2 根据trainNB0的结果，使用classifyNB来进行分类

# 根据样本数据集训练，分别得到p0和p1的条件概率
# trainMatrix---样本数据矩阵(m * n)
# trainCategory---样本数据的类别组成的列向量(n * 1)
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 样本个数
    numWords = len(trainMatrix[0])  # 样本的特征维数(垃圾邮件识别是常见词语的出现次数)
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # p1(1 表示属于类别1 p0表示不属于类别1 多元分类类似于logistic回归)
                                                       # pAbusive---属于类别1的样本占总样本数据的种类
                                                       # sum(trainCategory)计算的是样本数据中属于类别1的个数(trainCategory用0和1来标识)
    p0Num = ones(numWords); p1Num = ones(numWords)  # 将每种特征出现的次数初始化为1，防止次数为0时导致计算结果为0
                                                    # p0Num和p1Num是(1 * n)行向量
    p0Denom = 2.0; p1Denom = 2.0  # 将分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 如果属于类别1
            p1Num += trainMatrix[i]  # p1Num统计类别1的样本中的各特征出现的总次数
            p1Denom += sum(trainMatrix[i])  # 一个实数，统计类别1的样本中的特征出现的总次数
        else:
            p0Num += trainMatrix[i]  # p0Num统计类别0的样本中的各特征出现的总次数
            p0Denom += sum(trainMatrix[i])  # 一个实数，统计类别0的样本中的特征出现的总次数
    p1Vect = log(p1Num/p1Denom)          # 采用自然对数不会造成损失，与原函数增减区间相同，防止因概率太小相乘时产生的下溢
    p0Vect = log(p0Num/p0Denom)          # 采用自然对数不会造成损失，与原函数增减区间相同，防止因概率太小相乘时产生的下溢
    return p0Vect, p1Vect, pAbusive  # pAbusive(也就是pClass1)是类别1占总样本的比例
                                      # 1-pAbusive(也就是1.0 - pClass1)不是类别1占总样本的比例

# 分类
# 通过比较两种条件概率来判断输入样本的类别
# vec2Classify---待分类的输入样本
# p0Vec---(1 * n)行向量，样本属于类别0时，各特征出现总次数占特征总出现次数的概率
# p1Vec---(1 * n)行向量，样本属于类别1时，各特征出现总次数占特征总出现次数的概率
# pClass1---属于类别1的样本占总样本数据的种类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  #计算p1的条件概率，与计算p1Vect时取对数进行统一
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)  #计算p0的条件概率，与计算p0Vect时取对数进行统一
    if p1 > p0:    # 对两种条件概率进行比较，判断哪种概率更高
        return 1
    else: 
        return 0


