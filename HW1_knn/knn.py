import numpy
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import operator

# 从数据集中抽取训练集和验证集
def data_sampling():
    s = numpy.loadtxt('iris.txt', dtype=float, delimiter=",")  # 读取txt文件数据
    # 从三类中均匀取值
    s1 = s[0:50, :] # label=1
    s2 = s[50:100, :] # label=2
    s3 = s[100:150, :] # label=3
    x1_train, x1_test = train_test_split(s1, test_size=10)
    x2_train, x2_test = train_test_split(s2, test_size=10)
    x3_train, x3_test = train_test_split(s3, test_size=10)
    # 合并抽取出的训练集和验证集
    x_train = numpy.r_[x1_train, x2_train, x3_train]
    x_test = numpy.r_[x1_test, x2_test, x3_test]
    # 打乱训练集
    index = [i for i in range(120)]
    random.shuffle(index)
    train_set = x_train[index]
    # 打乱验证集
    index = [i for i in range(30)]
    random.shuffle(index)
    test_set = x_test[index]
    return train_set, test_set

# 计算模式特征的欧氏距离
def compute_distance(test, train, row):
    distance = 0
    for i in range(row):
        distance += (test[i] - train[i])**2
    return distance**0.5

# k近邻
def knn(train_set, test, k):
    # 计算测试集实例到训练集实例的欧式距离
    distances = [] # 存放训练集实例与距离
    for i in range(len(train_set)):
        dist = compute_distance(test, train_set[i], len(test)-1)
        distances.append((train_set[i], dist))
    #按距离由小到达进行排序
    distances.sort(key=operator.itemgetter(1))

    #在所有距离中取前k个，即k近邻
    neighbors = [] # 存放k近邻的训练集实例
    for i in range(k):
        neighbors.append(distances[i][0])

    #统计k近邻中各标签的频率
    label = {}
    for i in range(len(neighbors)):
        train_label = neighbors[i][4] # 训练集实例的标签
        if train_label in label:
            label[train_label] += 1
        else:
            label[train_label] = 1
    # 按标签频率降序排列
    result = sorted(label.items(), key = operator.itemgetter(1), reverse=True)
    return result[0][0] # 返回标签中的众数


def f():
    accuracy = [0] * 50

    # 多次验证
    for j in range(0, 5):
        # 训练集：测试集 = 8:2
        train_set, test_set = data_sampling()
        # k分别取值1~50进行训练测试
        for k in range(0, 50):
            for i in range(len(test_set)): # 验证测试集中每一条实例
                train_result = knn(train_set, test_set[i], k + 1)
                if train_result == test_set[i][4]: # 训练得到的结果与训练实例的标签是否一致
                    accuracy[k] = accuracy[k] + 1 # 若一致,准确度+1

    # 画图
    x = range(1, 51)  # x轴范围
    res = numpy.c_[x, accuracy]
    plt.scatter(res[:,0], res[:,1])  # 散点图
    plt.title("kNN")  # 图形标题
    plt.xlabel('k value')  # x轴名称
    plt.ylabel('accuracy')  # y轴名称
    plt.xticks(x)  # 设置x轴刻度
    plt.show()  # 显示图形

    # 找到准确率最高的k值
    max_accuracy = accuracy[0] / (30 * 5)
    for i in range(1, 50):
        accuracy[i] = accuracy[i] / (30 * 5)
        if max_accuracy < accuracy[i]:
            max_accuracy = accuracy[i]

    # 打印正确率最大的k值
    for i in range(0, 50):
        if max_accuracy == accuracy[i]:
            best_k = i + 1
            print('最优k值：', best_k)
    print('最优预测正确率：', max_accuracy * 100, "%")

    return best_k, max_accuracy

#统计结果
def main():
    f_k=[]
    f_accuracy=[]
    # 记录每次运行的结果
    for i in range(10):
        print('第', i+1, '次运行结果：')
        best_k, best_accuracy=f()
        f_k.append(best_k)
        f_accuracy.append(best_accuracy)
    # 计算均值
    k_mean = numpy.mean(f_k)
    accuracy_mean = numpy.mean(f_accuracy)
    print('----10次运行结果统计----')
    print('k的均值：', k_mean, '正确率的均值：', accuracy_mean* 100, "%")
    print()

if __name__ == "__main__":
    main()

