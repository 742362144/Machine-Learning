# -*- coding: utf-8 -*-
'''
Created on

@author: Belle
'''
from numpy.random.mtrand import randint
import numpy as np
import time

'''双曲函数'''


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


'''双曲函数的导数'''

@np.vectorize
def sigmoidDer(x):
    tanhValue = sigmoid(x)
    return tanhValue * (1 - tanhValue)


'''
Bp神经网络model
'''


class BpNeuralNetWorkModel:
    def __init__(self, inputLayerNumebr, outputLayerNumber, layerNumber, studyRate):
        '''学习率'''
        self.studyRate = studyRate
        self.inputLayerNumebr = inputLayerNumebr
        self.outputLayerNumber = outputLayerNumber
        self.layerNumber = layerNumber
        '''计算隐藏层神经元的数量'''
        # self.hiddenNeuronNum = int(np.sqrt(inputLayerNumebr + outputLayerNumber) + randint(1, 10))
        self.hiddenNeuronNum = 5
        '''层数据'''
        self.layers = []
        '''创建输出层'''
        inputLayer = Layer()
        inputLayer.initW(inputLayerNumebr+1, self.hiddenNeuronNum)
        self.layers.append(inputLayer)

        '''创建隐藏层, 并初始化权重'''
        for index in range(layerNumber - 2):
            currentLayer = Layer()
            '''初始化各个层的权重'''
            '''隐藏层到隐藏层'''
            if index == layerNumber - 3:
                currentLayer.initW(self.hiddenNeuronNum + 1, self.outputLayerNumber)
            else:
                currentLayer.initW(self.hiddenNeuronNum + 1, self.hiddenNeuronNum)
            self.layers.append(currentLayer)
        ''' 初始化输出层 '''
        outputLayer = Layer()
        self.layers.append(outputLayer)

    '''神经网络前向传播'''
    def forwardProp(self, x):
        '''计算输入层的输出值'''
        inputLayer = self.layers[0]
        inputLayer.alphas = np.row_stack((np.mat([1]), x))
        inputLayer.caculateOutPutValues()
        '''计算隐藏层的输出值'''
        for index in range(1, len(self.layers) - 1):
            hiddenLayer = self.layers[index]
            '''上一层的out put values就是这一层的zValues'''
            hiddenLayer.zValues = self.layers[index-1].outPutValues
            '''计算alphas'''
            hiddenLayer.caculateAlphas(isOutPut=False)
            '''输入层计算out puts'''
            hiddenLayer.caculateOutPutValues()
        '''计算输出层的alpha'''
        outputLayer = self.layers[-1]
        '''上一层的out put values就是这一层的zValues'''
        outputLayer.zValues = self.layers[-2].outPutValues
        '''计算alphas'''
        outputLayer.caculateAlphas(isOutPut=True)
        '''最后一层不需要求输出值，只要求出alpha'''

    '''神经网络后向传播'''
    def backProp(self, y):
        outputLayer = self.layers[-1]
        outputLayer.y = y

        '''输出层的残差值'''
        outputLayer.caculateOutPutLayerError()

        '''输出层到隐藏层'''
        preLayer = outputLayer
        layerCount = len(self.layers) - 1
        while layerCount >= 1:
            '''当前层'''
            currentLayer = self.layers[layerCount - 1]
            '''更新权重'''
            if layerCount != 1:
                currentLayer.computeDelta(preLayer.delta, self.studyRate)
            layerCount = layerCount - 1
            preLayer = currentLayer

    '''更新权重'''
    ''''''
    def updateWeight(self):
        for i in range(self.layerNumber-1):
            print(i)
            delta = self.layers[i+1].delta * self.layers[i].alphas.T
            self.layers[i].w = self.layers[i].w - self.studyRate * delta



'''
创建层
'''


class Layer:
    def __init__(self):
        self.bias = 1

    '''使用正态分布的随机值初始化w的值'''

    def initW(self, numOfAlpha, nextLayNumOfAlpha):
        self.w = np.mat(np.random.randn(nextLayNumOfAlpha, numOfAlpha))
    '''计算当前层的alphas'''

    def caculateAlphas(self, isOutPut):
        '''alpha = f(z) f为激活函数'''
        if isOutPut:
            self.alphas = sigmoid(self.zValues)
        else:
            self.alphas = np.row_stack((np.mat([1]), sigmoid(self.zValues)))
        # print(self.alphas)
    '''计算out puts'''

    def caculateOutPutValues(self):
        # 加上偏置单元
        '''计算当前层z = w * alpha的的下一层的输入值'''
        self.outPutValues = self.w * self.alphas

    '''计算输出层的残差'''
    def caculateOutPutLayerError(self):
        '''求f'(z)的值（即f的导数值）'''
        zDerValues = sigmoidDer(self.zValues)
        self.delta = np.multiply(self.alphas - self.y, zDerValues)

    '''计算其它层的Delta'''
    def computeDelta(self, preDelta, studyRate):
        zDerValues = sigmoidDer(self.zValues)
        self.delta = np.multiply(np.dot(self.w.T, preDelta)[1:, :], zDerValues)

'''
训练神经网络模型
@param train_set: 训练样本
@param labelOfNumbers: 训练总类别
@param layerNumber:  神经网络层数，包括输出层，隐藏层和输出层(默认只有一个输入层，隐藏层和输出层)
'''


def train(X, y, layerNumber=3, sampleTrainningTime=5000, studyRate=0.6):
    neuralNetWork = BpNeuralNetWorkModel(len(X[0]), len(y[0]), layerNumber, studyRate)
    '''训练数据'''
    for i in range(len(X)):
        '''当个样本使用梯度下降的方法训练sampleTrainningTime次'''
        for time in range(sampleTrainningTime):
            '''前向传播 '''
            neuralNetWork.forwardProp(np.mat(X[i]).T)
            '''反向传播'''
            neuralNetWork.backProp(np.mat(y[i]).T)
        neuralNetWork.updateWeight()

X = [[0.05, 0.1, 0.3], [0.3, 0.2, 0.4]]
y = [[0.1, 0.3, 0.122, 0.1], [0.99, 0.3, 0.122, 0.9]]
layerOfNumber = 4

train(X, y, layerOfNumber)