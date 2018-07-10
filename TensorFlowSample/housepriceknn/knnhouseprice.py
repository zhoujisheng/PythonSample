import numpy as np
import tensorflow as tf
import pandas as pd

data_path = "dataset/king_county_data_geocoded.csv"

# 从csv里读取数据到dataframe里去，告诉pandas第一行是标题，另外只要限定的4列
data = pd.read_csv(data_path, header=0)[['AppraisedValue', 'SqFtLot', 'lat', 'long']]

# 构建样本数据和测试数据
train_data = data[0:50000]
test_data = data[50000:51000]

# pop的含义是将指定列从数据集中抽出来
train_x, train_y = train_data, train_data.pop('AppraisedValue')
test_x, test_y = test_data, test_data.pop('AppraisedValue')

# 归一化，经过调试发现z-score的效果更好。min-max差一点
# train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
# test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())
train_x = (train_x - train_x.mean()) / (train_x.std())
test_x = (test_x - test_x.mean()) / (test_x.std())

# 定义占位符，tensorflow run的时候用到
xtr = tf.placeholder("float", [None, 3])
xte = tf.placeholder("float", [3])
# 使用曼哈顿距离，测试欧氏距离效果差一点
# reduce_sum是降维求和的方法，reduction_indices=1 代表按行求和
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
#distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr,tf.negative(xte)),2),reduction_indices=1))
# 取出距离最近的100个点
pred = tf.nn.top_k(tf.negative(distance),50)
accuracy = 0.
# tensorflow需要的初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 循环，每次训练集全取，测试集取一行
    for i in range(len(test_x)):
        nn_index = sess.run(pred, feed_dict={xtr: train_x, xte: test_x.iloc[i]})
        # 最近的几个点求均值作为预测值
        pre_value = np.mean(train_y[nn_index.indices])
        print("Test", i, "预测值:", pre_value,"真实值:", test_y.iloc[i])
        # 偏差10%以内认为可以接受
        if abs(pre_value /  test_y.iloc[i]-1)<=0.1:
            accuracy += 1. / len(test_x)
    print("结束!")
    print("准确率:", accuracy)