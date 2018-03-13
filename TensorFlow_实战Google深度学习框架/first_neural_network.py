# @Time    : 2018/3/13 17:58
# @Author  : Leafage
# @File    : first_neural_network.py
# @Software: PyCharm
# @Describe: 采用模拟数据训练神网络
import tensorflow as tf
from numpy.random import RandomState

# 定以训练数据batch的大小，每轮训练都会选取这么多的数据进行训练
batch_size = 8

# 定以神经网络的参数，使用2*3和3*1的矩阵形式
# 定义一个2*3的矩阵，标准差为1，随机数种子为1
# 也就是第一层神经元的参数变量
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# 第二层神经元的参数变量
# 定义一个3*1的矩阵，标准差为1，随机数种子为1
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 占位符，避免使用太多常量从而消费时间和资源,这个位置中的数据在程序运行的时候再指定
# 这里指定了参数的类型，shape中的Node代表任意都可以，也就是说该矩阵形状是两列的，但是行数不一定
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 对应样本的标签
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络向前传播的过程
# 这里会执行矩阵相乘的过程，所以y的值就是x和各个参数相乘得到的
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
# 随机数中子设置为1
rdm = RandomState(1)
# 设定矩阵的行数
dataset_size = 128
# 生成一个随机的128*2的矩阵
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签。只要x1+x2<1就认为是正样本（用1表示正样本），其他的为负样本（用0表示负样本）
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行程序
with tf.Session() as sess:
    # 初始化所有变量，这里也就是生成w1和w2
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 计算w1和w2原始的数值
    print('--------------------\n训练之前的参数：')
    print('w1:')
    print(sess.run(w1))
    print('w2:')
    print(sess.run(w2))

    # 定义训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        # 结束的位置不能超过样本总和
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})

        # 每隔一段时间计算在所有数据上的交叉熵并输出
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print('第%d次迭代，此时交叉熵为：%g' % (i, total_cross_entropy))

    # 训练之后的神经网络参数的值
    print('----------------\n训练之后的参数：')
    print('w1:')
    print(sess.run(w1))
    print('w2:')
    print(sess.run(w2))

