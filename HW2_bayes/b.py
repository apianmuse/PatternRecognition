import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils.extmath import fast_logdet
import math


def main():
    # 第1问 #
    N = 1000
    # 均值
    m1 = np.array([1, 1])
    m2 = np.array([4, 4])
    m3 = np.array([8, 1])
    # 协方差矩阵
    v = np.array([[2, 0], [0, 2]])
    # 生成数据集合X(来自三个分布模型的先验概率Pr(wi)相同1/3)
    n1 = N // 3
    n2 = N // 3
    n3 = N - n1 - n2
    x1 = np.random.multivariate_normal(m1, v, n1)
    x2 = np.random.multivariate_normal(m2, v, n2)
    x3 = np.random.multivariate_normal(m3, v, n3)
    # 生成数据集合X'(先验概率分别为0.6、0.3、0.1)
    nn1 = int(N * 0.6)
    nn2 = int(N * 0.3)
    nn3 = int(N * 0.1)
    xx1 = np.random.multivariate_normal(m1, v, nn1)
    xx2 = np.random.multivariate_normal(m2, v, nn2)
    xx3 = np.random.multivariate_normal(m3, v, nn3)

    第2问 #
    分别画出所生成的两个数据集合中随机矢量的散布图
    plt.figure(0)
    plt.title('Set X')
    plt.plot(x1[:, 0], x1[:, 1], '.r', x2[:, 0], x2[:, 1], '.g', x3[:, 0], x3[:, 1], '.b')
    plt.figure(1)
    plt.title("Set X'")
    plt.plot(xx1[:, 0], xx1[:, 1], '.r', xx2[:, 0], xx2[:, 1], '.g', xx3[:, 0], xx3[:, 1], '.b')
    plt.show()

    # 第3问 #

    # 正态分布概率密度函数-似然函数p(x|wi)
    def gauss(x, m, s):
        t = x - m
        return np.exp(-1 / 2 * np.dot(np.dot(np.transpose(t), np.linalg.inv(s)), t))

    # 似然率测试规则LRT #
    def lrt(x, s, p, j, n):
        # 数据x, 均值m, 协方差s, 先验概率p, 模式j是正确的, 第j组的个数n
        # 返回：第j组的错误个数
        e = 0
        for i in list(range(0, n)):
            g1 = p * gauss(x[i], m1, s)
            g2 = p * gauss(x[i], m2, s)
            g3 = p * gauss(x[i], m3, s)
            if g1 > g2 and g1 > g3:
                plt.plot(x[i][0], x[i][1], '.r')
                if j != 1:
                    e = e + 1
            elif g1 < g2 and g3 < g2:
                plt.plot(x[i][0], x[i][1], '.g')
                if j != 2:
                    e = e + 1
            else:
                plt.plot(x[i][0], x[i][1], '.b')
                if j != 3:
                    e = e + 1
        return e

    # 数据集合X
    plt.figure(2)
    plt.title('Set X & LRT')
    l_e1 = lrt(x1, v, 1 / 3, 1, n1)
    l_e2 = lrt(x2, v, 1 / 3, 2, n2)
    l_e3 = lrt(x3, v, 1 / 3, 3, n3)
    plt.show()
    # 错误率
    l_error = (l_e1 + l_e2 + l_e3) / N
    print('X & 似然率测试规则 分类错误率:', l_error * 100, "%")

    # 数据集合X'
    plt.figure(3)
    plt.title("Set X' & LRT")
    l_e1 = lrt(xx1, v, 0.6, 1, nn1)
    l_e2 = lrt(xx2, v, 0.3, 2, nn2)
    l_e3 = lrt(xx3, v, 0.1, 3, nn3)
    plt.show()
    # 错误率
    l_error = (l_e1 + l_e2 + l_e3) / N
    print("X' & 似然率测试规则 分类错误率:", l_error * 100, "%")

    # 最小化贝叶斯风险决策规则 #
    def bayes(x, s, p, j, n):
        # 数据x, 协方差s, 先验概率p, 模式j是正确的, 第j组的个数n
        # 返回：第j组的错误个数
        e = 0
        for i in list(range(0, n)):
            g1 = 0 * gauss(x[i], m1, s) * p + 2 * gauss(x[i], m2, s) * p + 3 * gauss(x[i], m3, s) * p
            g2 = 1 * gauss(x[i], m1, s) * p + 0 * gauss(x[i], m2, s) * p + 2.5 * gauss(x[i], m3, s) * p
            g3 = 1 * gauss(x[i], m1, s) * p + 1 * gauss(x[i], m2, s) * p + 0 * gauss(x[i], m3, s) * p
            if g1 < g2 and g1 < g3:
                plt.plot(x[i][0], x[i][1], '.r')
                if j != 1:
                    e = e + 1
            elif g2 < g1 and g2 < g3:
                plt.plot(x[i][0], x[i][1], '.g')
                if j != 2:
                    e = e + 1
            else:
                plt.plot(x[i][0], x[i][1], '.b')
                if j != 3:
                    e = e + 1
        return e

    # 数据集合X
    plt.figure(4)
    plt.title('Set X & bayes')
    b_e1 = bayes(x1, v, 1 / 3, 1, n1)
    b_e2 = bayes(x2, v, 1 / 3, 2, n2)
    b_e3 = bayes(x3, v, 1 / 3, 3, n3)
    plt.show()
    # 错误率
    b_error = (b_e1 + b_e2 + b_e3) / N
    print('X & 最小化贝叶斯风险决策规则 分类错误率:', b_error * 100, "%")

    # 数据集合X'
    plt.figure(5)
    plt.title("Set X' & bayes")
    b_e1 = bayes(xx1, v, 0.6, 1, nn1)
    b_e2 = bayes(xx2, v, 0.3, 2, nn2)
    b_e3 = bayes(xx3, v, 0.1, 3, nn3)
    plt.show()
    # 错误率
    b_error = (b_e1 + b_e2 + b_e3) / N
    print("X' & 最小化贝叶斯风险决策规则 分类错误率:", b_error * 100, "%")

    # 最大后验概率决策规则 #
    def mmap(x, s, p, j, n):
        # 数据x, 均值m, 协方差s, 先验概率p, 模式j是正确的, 第j组的个数n
        # 返回：第j组的错误个数

        # 判别式函数
        def map_gauss(_x, _m, _s, _p):
            t = _x - _m
            return -1 / 2 * np.dot(np.dot(np.transpose(t), np.linalg.inv(_s)), t) - 1 / 2 * fast_logdet(_s) + math.log(
                _p, math.e)

        e = 0
        for i in list(range(0, n)):
            g1 = p * map_gauss(x[i], m1, s, p)
            g2 = p * map_gauss(x[i], m2, s, p)
            g3 = p * map_gauss(x[i], m3, s, p)
            if g1 > g2 and g1 > g3:
                plt.plot(x[i][0], x[i][1], '.r')
                if j != 1:
                    e = e + 1
            elif g1 < g2 and g3 < g2:
                plt.plot(x[i][0], x[i][1], '.g')
                if j != 2:
                    e = e + 1
            else:
                plt.plot(x[i][0], x[i][1], '.b')
                if j != 3:
                    e = e + 1
        return e

    # 数据集合X
    plt.figure(2)
    plt.title('Set X & MAP')
    m_e1 = mmap(x1, v, 1 / 3, 1, n1)
    m_e2 = mmap(x2, v, 1 / 3, 2, n2)
    m_e3 = mmap(x3, v, 1 / 3, 3, n3)
    plt.show()
    # 错误率
    m_error = (m_e1 + m_e2 + m_e3) / N
    print('X & 最大后验概率决策规则 分类错误率:', m_error * 100, "%")

    # 数据集合X'
    plt.figure(3)
    plt.title("Set X' & MAP")
    m_e1 = mmap(xx1, v, 0.6, 1, nn1)
    m_e2 = mmap(xx2, v, 0.3, 2, nn2)
    m_e3 = mmap(xx3, v, 0.1, 3, nn3)
    plt.show()
    # 错误率
    m_error = (m_e1 + m_e2 + m_e3) / N
    print("X' & 最大后验概率决策规则 分类错误率:", m_error * 100, "%")

    # 最短欧氏距离规则 #
    def ed(x, j, n):
        # 数据x, 模式j是正确的, 第几组的个数n
        # 返回：第j组的错误个数
        e = 0
        for i in list(range(0, n)):
            t1 = x[i] - m1
            t2 = x[i] - m2
            t3 = x[i] - m3
            g1 = np.dot(t1, np.transpose(t1))
            g2 = np.dot(t2, np.transpose(t2))
            g3 = np.dot(t3, np.transpose(t3))
            if g1 < g2 and g1 < g3:
                plt.plot(x[i][0], x[i][1], '.r')
                if j != 1:
                    e = e + 1
            elif g2 < g1 and g2 < g3:
                plt.plot(x[i][0], x[i][1], '.g')
                if j != 2:
                    e = e + 1
            else:
                plt.plot(x[i][0], x[i][1], '.b')
                if j != 3:
                    e = e + 1
        return e

    # 数据集合X
    plt.figure(6)
    plt.title('Set X & Euclidean Distance')
    e_e1 = ed(x1, 1, n1)
    e_e2 = ed(x2, 2, n2)
    e_e3 = ed(x3, 3, n3)
    plt.show()
    # 错误率
    e_error = (e_e1 + e_e2 + e_e3) / N
    print('X & 最短欧氏距离规则 分类错误率:', e_error * 100, "%")

    # 数据集合X'
    plt.figure(7)
    plt.title("Set X' & Euclidean Distance")
    e_e1 = ed(xx1, 1, nn1)
    e_e2 = ed(xx2, 2, nn2)
    e_e3 = ed(xx3, 3, nn3)
    plt.show()
    # 错误率
    e_error = (e_e1 + e_e2 + e_e3) / N
    print("X' & 在最短欧氏距离规则 分类错误率:", e_error * 100, "%")


if __name__ == '__main__':
    for k in range(5):
        print("第", k, "次实验结果：")
        main()
        print()
