from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


class KNN(object):
    def __init__(self):
        print("hello KNN!")


    def file2matrix(self, filename):
        fr = open(filename)
        numberOfLines = len(fr.readlines())  # get the number of lines in the file
        returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
        classLabelVector = []  # prepare labels return
        fr = open(filename)
        # 文件格式为四列 前三列位数字 后一列为名字
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector

    def draw(self, datingDataMat):
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


knn = KNN()
datingDataMat, datingLabels = knn.file2matrix('datingTestSet2.txt')
print(datingDataMat, datingLabels)
knn.draw(datingDataMat)
