# # =============神经网络用于回归=============

import numpy as np
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler
data = [
         [ -0.017612,14.053064,14.035452],[ -1.395634, 4.662541, 3.266907],[ -0.752157, 6.53862,5.786463],[ -1.322371, 7.152853, 5.830482],
         [0.423363,11.054677,11.47804 ],[0.406704, 7.067335, 7.474039],[0.667394,12.741452,13.408846],[ -2.46015,6.866805, 4.406655],
         [0.569411, 9.548755,10.118166],[ -0.026632,10.427743,10.401111],[0.850433, 6.920334, 7.770767],[1.347183,13.1755,14.522683],
         [1.176813, 3.16702,4.343833],[ -1.781871, 9.097953, 7.316082],[ -0.566606, 5.749003, 5.182397],[0.931635, 1.589505, 2.52114 ],
         [ -0.024205, 6.151823, 6.127618],[ -0.036453, 2.690988, 2.654535],[ -0.196949, 0.444165, 0.247216],[1.014459, 5.754399, 6.768858],
         [1.985298, 3.230619, 5.215917],[ -1.693453,-0.55754, -2.250993],[ -0.576525,11.778922,11.202397],[ -0.346811,-1.67873, -2.025541],
         [ -2.124484, 2.672471, 0.547987],[1.217916, 9.597015,10.814931],[ -0.733928, 9.098687, 8.364759],[1.416614, 9.619232,11.035846],
         [1.38861,9.341997,10.730607],[0.317029,14.739025,15.056054]
]

dataMat = np.array(data)
X=dataMat[:,0:2]
y = dataMat[:,2]
scaler = StandardScaler() # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集

# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
print('预测结果：', clf.predict([[0.317029, 14.739025]]))  # 预测某个输入对象

cengindex = 0
for wi in clf.coefs_:
    cengindex += 1  # 表示底第几层神经网络。
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:',wi.shape)
    print('系数矩阵：\n',wi)