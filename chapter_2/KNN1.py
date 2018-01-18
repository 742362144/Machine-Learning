from numpy import *
import operator

class KNN(object):

    def __init__(self):
        print("hello KNN!")

    def createDataSet(self):
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.0, 1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels

    def classify0(self, inx, dataSet, labels,k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inx, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlable = labels[sortedDistIndicies[i]]
            classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

knn = KNN()
group, labels = knn.createDataSet()
print(knn.classify0([0,0], group, labels, 3))