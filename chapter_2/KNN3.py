from numpy import *
from os import listdir
from chapter_2 import KNN2

# 手写识别系统
# 开销太大，使用k决策树

class KNNWrite(object):

    def __init__(self):
        self.knn = KNN2.KNN()

    def img2Vector(self, filename):
        returnVect = zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i + j] = int(lineStr[j])
        return returnVect

    def handWritingClassTest(self):
        hwLabels = []
        trainingFileList = listdir('trainingDigits')
        # 获取目录下总文件数
        m = len(trainingFileList)
        # 新建 m×1024 大小的矩阵
        trainingMat = zeros((m, 1024))
        # 将训练数据文件中的数据读入矩阵
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i, :] = self.img2Vector('trainingDigits/%s' % fileNameStr)
        # 获取待测试文件列表
        testFileList = listdir('testDigits')
        errorCount = 0
        mTest = len(testFileList)
        # 依次测试测试文件夹的所有文件
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = self.img2Vector('testDigits/%s' % fileNameStr)
            classifierResult = self.knn.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
            print("the classifierResult came back with: %d, the real answer is: %d"
                  % (classifierResult, classNumStr))
            if(classifierResult != classNumStr) :
                errorCount += 1.0
        print("\nthe total number of errors is %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

if __name__ == "__main__":
    knn = KNNWrite()
    # testVector = knn.img2Vector('testDigits/0_14.txt')
    # print(testVector)
    knn.handWritingClassTest()