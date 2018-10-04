import numpy as np


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes

        self.no_of_hidden_nodes = no_of_hidden_nodes

        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural 
        network with optional bias nodes"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        # 创建大小为
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # 输入层到隐藏层的权值
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes + bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # 隐藏层到输出层的权值
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        # 前向传播
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        # 隐藏层的输出向量
        output_vector_hidden = activation_function(output_vector1)

        if self.bias:
            # 为下一层的输入加上偏置单元
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        # 输出层的输出值
        output_vector_network = activation_function(output_vector2)

        # 反向传播
        # 计算输出层的残差
        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        # tmp为 S3 * 1 维列向量
        # output_vector_hidden.T 是 1 * (S2 + 1)维行向量
        # tmp的维数是 S3 * (S2 + 1)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp
        # calculate hidden errors:
        # 计算隐藏层的残差
        # self.weights_hidden_out.T 为 (S2 + 1) * S3
        #  output_errors 为 S3 * 1
        # hidden_errors 为 (S2 + 1) * 1
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        ''' 
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
            因为output_vector_hidden中bias为1, 则在1.0 - output_vector_hidden 中bias所在单元变为0
        '''
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        # 反向传播时2去掉最后一行(也就是偏置所在的行)
        # tmp 为 (S2 + 1) * 1
        # input_vector.T 为 1 * (S1 + 1)
        if self.bias:
            # print(tmp)
            # print(tmp.shape)
            # print(input_vector.T.shape)
            # x = np.dot(tmp[:-1, :], input_vector.T)  # ???? last element cut off, ???
            x = np.dot(tmp, input_vector.T)[:-1, :]  # ???? last element cut off, ???
            # print(x.shape)
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x

    def predict(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector


class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3)]
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8)]
labeled_data = []
for el in class1:
    labeled_data.append([el, [1, 0]])
for el in class2:
    labeled_data.append([el, [0, 1]])

np.random.shuffle(labeled_data)
# print(labeled_data[:10])
data, labels = zip(*labeled_data)
labels = np.array(labels)
data = np.array(data)

simple_network = NeuralNetwork(no_of_in_nodes=2,
                               no_of_out_nodes=2,
                               no_of_hidden_nodes=10,
                               learning_rate=0.1,
                               bias=1)

for _ in range(100):
    for i in range(len(data)):
        simple_network.train(data[i], labels[i])
# for i in range(len(data)):
#     print(labels[i])
#     print(simple_network.predict(data[i]))