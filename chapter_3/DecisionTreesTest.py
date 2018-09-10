from math import log
import operator

# 原则----将无序的数据变得更加有序
# 计算每个特征值划分数据集所获得信息增益
# 获得信息增益最高的特征就是最好的选择
# 计算信息增益也就是计算香农熵
from chapter_3.DecisionTrees import DecisionTrees


class DeciionTreesTest(object):

    # 人为创造数据
    def createDataSet(self):
        dataSet = [
            [1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']
        ]
        labels = ['no surfacing', 'flippers']
        return dataSet, labels



if __name__ == "__main__":
    test = DeciionTreesTest()
    tree = DecisionTrees()
    myDat, labels = tree.createDataSet()
    # tree.splitDataSet(myDat, 0, 1)
    # print(tree.chooseBestFeatureToSplit(myDat))
    print(tree.createTree(myDat, labels))