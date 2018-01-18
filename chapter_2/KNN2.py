from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


# 这个例子是根据海伦以前交往的三种类型的人的数据作为训练集
# 测试的数据文件的最后一行为以前交往的对象类型（1000个也是厉害了）1:no-like 2:normal 3:like
# 来预测要测试的输入数据是什么类型

class KNN(object):
    def __init__(self):
        print("hello KNN!")

    def file2matrix(self, filename):
        fr = open(filename)
        numberOfLines = len(fr.readlines())  # get the number of lines in the file
        # zeros----返回 m×n×p×...的double类零矩阵
        returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
        # 约会对象的类型，有三种类型分别为 1 2 3
        classLabelVector = []  # prepare labels return
        fr = open(filename)
        # 文件格式为四列 前三列位数字 后一列为名字
        index = 0
        for line in fr.readlines():
            # 移除字符串头尾指定的字符（默认为空格）
            line = line.strip()
            # 移除tab制表符，文件中每行数据用\t分隔
            listFromLine = line.split('\t')
            # returnMat----1000×3的矩阵
            returnMat[index, :] = listFromLine[0:3]
            # 读取每一行将每一行扔进returnMat
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector
    # k邻近算法----采用欧式距离公式
    # inx----用于分类的输入向量，也就是要测试的输入数据集
    # dataSet----输入的训练样本集
    # 标签向量----每个数据样本的标签，有三种 1， 2， 3
    # 选择最近邻居的数目，也就是选前几个最像的
    def classify0(self, inx, dataSet, labels, k):
        # 取数据的行数
        dataSetSize = dataSet.shape[0]
        # tile(inx, (dataSetSize, 1))--返回一个1000×3的矩阵，其中每一行都是inx
        # diffMat--dataSet的每一行整体减去inx所产生的数组
        diffMat = tile(inx, (dataSetSize, 1)) - dataSet
        # sqDiffMat等于diffMat的元素整体平方
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        # sqlDiffMat的每行数据相加，算出每行数据与inx的欧氏距离
        distances = sqDistances ** 0.5
        # distances排序
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
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        print(sortedClassCount)
        return sortedClassCount[0][0]

    # filename----存有数据的文本名
    # 文件格式为四列 前三列位数字 后一列为
    # 文件有1000行

    def draw(self, datingDataMat, datingLabels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 没有分色
        # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
        # 分色 使用第二三列
        # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
        #            15.0 * array(datingLabels), 15.0 * array(datingLabels))
        # 分色 使用第一二列 更为清晰
        ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
                   15.0 * array(datingLabels), 15.0 * array(datingLabels))
        plt.show()

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
        return normDataSet, ranges, minVals

    def datingClassTest(self):
        # 前90%作为训练样本 取后10%测试分类器
        # 注意取值，训练数据较少时
        hoRatio = 0.10
        # 从文件获取数据
        datingDataMat, datingLabels = self.file2matrix('datingTestSet2.txt')
        # 规格化数据
        normMat, ranges, minVals = self.autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVecs = int(m * hoRatio)
        errorCount = 0.0
        # 求每组数据与样本的欧氏距离
        for i in range(numTestVecs):
            classifierResult = self.classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
            # 看预测结果与实际的比较
            if (classifierResult != datingLabels[i]):
                errorCount += 1.0
            break  # todo
        print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
        print(errorCount)

knn = KNN()
# datingDataMat, datingLabels = knn.file2matrix('datingTestSet2.txt')
#
# knn.draw(datingDataMat, datingLabels)

# normMat, ranges, minVals = knn.autoNorm(datingDataMat)
knn.datingClassTest()
