from math import log
import operator

# TODO 使用说明
# TODO 1 格式化样本数据
# TODO 2 通过样本数据构建决策树
# TODO 3 输入样本数据通过决策树进行分类

# 原则----将无序的数据变得更加有序
# 计算每个特征值划分数据集所获得信息增益
# 获得信息增益最高的特征就是最好的选择
# 计算信息增益也就是计算香农熵
class DecisionTrees(object):

    def __init__(self):
        print("hello DecisionTrees!")

    # 计算给定数据的香农熵
    # 然后可以根据获取最大增益的方法来划分数据集
    # 注：也可以采用基尼不纯度（Gini impurity）
    # dataSet----样本数据集
    def calcShannonEnt(self, dataSet):
        # 总共的数据样本个数
        numEntries = len(dataSet)
        # 存储 [{分类 : 样本个数}] 这样的键值对
        labelCounts = {}
        # 为所有分类创建字典
        # 统计每个分类的样本个数
        for featVec in dataSet:
            # 获得当前的样本类别
            currentLabel = featVec[-1]
            # 如果种类标签没有添加进去， 添加进currentLabel并遍历统计频率
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannnonEnt = 0.0
        for key in labelCounts:
            # 计算这个类标签占总数据样本的比例
            prob = float(labelCounts[key])/numEntries
            # 以二为底求对数
            # 取负的原因是prob(分类占总数据样本的比例)<1，取对数后为负，故取负得正
            shannnonEnt -= prob * log(prob, 2)
        return shannnonEnt

    # 按照给定特征值划分数据
    # dataSet----样本数据集
    # axis----划分数据集的特征
    # value----需要返回的特征的值
    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []  # 新建列表，防止在函数声明周期修改dataSet，存储满足要求的样本(需要将样本的第axis剔除)
        for featVec in dataSet:
            # featVec为一个样本
            # 如果featVec的第axis个特征值 == value
            # 则将featVec的第axis特征值剔除，并将剔除featVec[axis]的形成的新向量添加进retDataSet
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)  # 将符合条件的数据添加入retDataSet
        return retDataSet

    # 选择最好的数据集划分方式
    # 遍历整个数据集，循环计算香农熵和splitDataSet()函数
    # dataSet----样本数据集
    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1  # 获得特征个数
        baseEntropy = self.calcShannonEnt(dataSet)  # 计算香农熵
        bestInfoGain = 0.0  # 最好的信息增益 初始为0
        bestFeature = -1  # 最好的划分特征 初始为-1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]  # 获得dataset的第i列
            uniqueVals = set(featList)  # 获得dataset的第i列的所有取值的种类
            newEntropy = 0.0
            for value in uniqueVals:  # 计算对第i个特征进行划分时的的信息熵
                # 根据第i个特征可能的取值进行划分
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy -= prob * self.calcShannonEnt(subDataSet)  # 取负的原因是prob(分类占总数据样本的比例)<1，取对数后为负，故取负得正
            infoGain = newEntropy - baseEntropy  # 计算信息增益
            if(infoGain > bestInfoGain):  # 找出信息增益最大的特征
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    # 此函数只在最后一次分类时使用
    # 当使用了所有的特征，仍无法将所有的数据集划分为仅含唯一类别的分组时，挑选出现次数最多的类别作为返回值
    # classList----样本数据集(classList为dataSet最后一列(也就是样本标签组成的列向量))
    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    # 创建决策树
    # 决策时的存储形式为键值对，字典嵌套的形式
    # dataSet----样本数据集
    # labels----每个特征所代表的实际意义(比如身高、体重等)
    def createTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):  # 判断classList中所有分类是否相等
            return classList[0]  # 当所有分类都相等时停止划分
        if len(dataSet[0]) == 1:  # 当只有一个分类(也就是只有类别组成的列向量)时停止划分
            return self.majorityCnt(classList)  # 将出现次数最多的类别作为返回值
        bestFeat = self.chooseBestFeatureToSplit(dataSet)  # 选择最好的划分特征
        bestFeatLabel = labels[bestFeat]  # 获取最好的划分特征标签
        myTree = {bestFeatLabel: {}}  # 决策树的存储形式
        del (labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)  # 获取最佳分类特征的所有取值
        for value in uniqueVals:
            subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
            retDataSet = self.splitDataSet(dataSet, bestFeat, value)  # 根据特征对数据集进行划分
            myTree[bestFeatLabel][value] = self.createTree(retDataSet, subLabels)  # 对划分好的数据集递归的创建决策树
        return myTree

    # 使用决策树进行分类
    # featLabels---每个特征所代表的实际意义(比如身高、体重等)
    def classify(self, inputTree, featLabels, testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):  # 如果key对应的value是dict的话说明还未到叶子节点，继续判断
            classLabel = self.classify(valueOfFeat, featLabels, testVec)
        else:
            classLabel = valueOfFeat  # 如果key对应的value不是dict，说明已经到达叶子节点，可以得出分类结论了
        return classLabel