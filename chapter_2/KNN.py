from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


# TODO 使用说明
# TODO 1 先使用autoNorm函数将样本数据归一化
# TODO 2 再将归一化后的dataSet和输入样本(使用autoNorm返回的ranges和minVals归一化)带入分类器

class KNN(object):
    def __init__(self):
        print("hello KNN!")

    # k邻近算法----采用欧式距离公式
    # inx----用于分类的输入向量，也就是要测试的输入样本
    # dataSet----输入的训练样本集
    # labels----标签向量，每个样本的标签组成的向量
    # k----选择最近邻居的数目，也就是选前几个最像的

    def classify0(self, inx, dataSet, labels, k):
        # 获取样本的的个数
        dataSetSize = dataSet.shape[0]
        # tile(inx, (dataSetSize, 1))--返回一个1000×3的矩阵，其中每一行都是inx
        # diffMat--dataSet的每一行整体减去inx所产生的数组
        diffMat = tile(inx, (dataSetSize, 1)) - dataSet
        # sqDiffMat等于diffMat的元素整体平方
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        # sqlDiffMat的每行数据相加，算出每行数据与inx的欧氏距离
        distances = sqDistances ** 0.5
        # 根据distances的大小进行排序
        # y = x.argsort()
        # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # 注意是下标
        sortedDistIndicies = distances.argsort()
        classCount = {}
        # 确定前k个距离最小的元素所在的主要分类
        for i in range(k):
            voteIlable = labels[sortedDistIndicies[i]]
            classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        # 对结果排序
        # 调用sorted(a)，对a进行排序后返回一个新的列表，而对a不产生影响
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    # 归一化特征值 ps:见书p25
    # 处理不同取值范围的特征值
    # 公式---- newValue = (oldValue-min)/(max-min)
    # dataSet----1000×3的二维矩阵
    def autoNorm(self, dataSet):
        # 取当前列的最小值
        minVals = dataSet.min(0)
        # 取当前列的最大值
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        # 创建一个和dataSet同规格的0矩阵(1000×3)
        normDataSet = zeros(shape(dataSet))
        # 获取dataSet的行数m
        m = dataSet.shape[0]
        # tile(minVals, (m, 1)) 创建了一个1000×3的矩阵
        # 且每一行的第三列为minVals
        # normDataSet等于dataSet的第三列整体减去minVals
        normDataSet = dataSet - tile(minVals, (m, 1))
        # tile(ranges, (m, 1)) 创建了一个1000×3的矩阵
        # 且每一行的第三列为ranges(ps:maxVals-minVals)
        # 相当于normSet的第三列整体除ranges
        normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
        # normDataSet是归一化之后的样本数据集
        # ranges, minVals 用来归一化待分类的输入向量
        return normDataSet, ranges, minVals

