import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------
#通过基尼系数选择最佳的分割点函数
def getBestPoint_Gini(data_x,data_y,data_index):
    ave_y = np.mean(data_y)     #获取数据标签的平均值作为标签的分割值
    Gini_index = []     #保存data_x数据中每个类型的最小基尼系数的位置索引
    T_index = []        #保存最小基尼系数的分割点
    ChildTree_group = []        #保存最优分割点前后的集合，保存data_index中的值
    for property in data_x:
        property = pd.DataFrame({"property":property,"data_y":data_y,"index":data_index})   #将标签和当前属性以及数据编号拼接在一起，以便于后面排序
        #print(property)
        #print(len(property["data_y"][1:][property["data_y"]<ave_y]))
        property=property.sort_values(by="property")    #将属性的数据排序
        T = (property["property"].values[0:len(property)-1]+property["property"].values[1:])/2        #获取每个属性的所有相邻两个数据的中位点数据，长度为len(data_x)-1
        index = property["index"].values    #获取排好序之后的标号数组
        ###注意，此时T已经转换为numpy的array对象
        Gini = []       #基尼系数数组
        T_min = []      #分割点数组
        group = []      #分割点前后的集合

        for i in range(len(T)):     #将每个中位点作为分割点求解其基尼系数，从中选择最小基尼系数得作为最优分割点
            smaller_lessNum = len(property["data_y"][:i+1][property["data_y"]<ave_y])   #统计分割点前的对应标签值小于平均标签值的个数
            bigger_lessNum = len(property["data_y"][i+1:][property["data_y"]<ave_y])      #统计分割点后的对应标签值大于等于平均标签值的个数
            smaller_p = smaller_lessNum / (i + 1)  # 获取分割点前数据对应小于当前分割点的数据的概率
            bigger_p = bigger_lessNum / (len(property)-i-1)
            gini = ((i+1)/len(property))*(1-smaller_p*smaller_p-(1-smaller_p)*(1-smaller_p)) + (1-(i+1)/len(property))*(1-bigger_p*bigger_p-(1-bigger_p)*(1-bigger_p))      #获取当前的基尼系数
            Gini.append(gini)    #将当前基尼系数保存
            T_min.append(T[i])      #将当前分割点保存
            group.append([index[:i+1],index[i+1:]])
        #print(Gini)

        Gini_index.append(np.min(Gini))     #若有两个及以上最小值时取第一个
        T_index.append(T_min[np.argmin(Gini)])
        ChildTree_group.append(group[np.argmin(Gini)])

    return Gini_index,T_index,ChildTree_group       #返回每个属性的最小基尼系数和对应分割点,以及分割点分割开的两边集合

#-----------------------------
#获取划分属性函数
def getBestProperty(data_x,data_y,data_index):

    Gini_index, T_index, ChildTree_group = getBestPoint_Gini(data_x,data_y,data_index)
    # print(Gini_index)
    bestProperty = np.argmin(Gini_index)    #最佳划分属性
    bestProperty_gini = np.min(Gini_index)      #基尼系数最小的为最优划分属性
    bestProperty_t = T_index[np.argmin(Gini_index)]     #获取最优划分属性的最优分割点

    left_index = ChildTree_group[np.argmin(Gini_index)][0]      #获取小于划分属性的分割点的集合
    left_data_x = data_x.T[left_index].T        #属性数据
    #left_data_x = np.delete(left_data_x,np.argmin(Gini_index),0)
    left_data_y = data_y[left_index]        #标签

    right_index = ChildTree_group[np.argmin(Gini_index)][1]      #获取大于划分属性分割点的集合
    right_data_x = data_x.T[right_index].T      #属性数据
    #right_data_x = np.delete(right_data_x,np.argmin(Gini_index),0)
    right_data_y = data_y[right_index]      #标签

    return [bestProperty_gini,bestProperty_t,bestProperty],left_data_x,left_data_y,right_data_x,right_data_y

# -----------------------------
# 生成决策树函数，递归生成
def creatTree(data_x, data_y, depth):
    tree = {}  # 决策树

    y = np.sort(data_y)
    if y[0]==y[-1] or len(data_y)==1:
        tree["value"] = y[0]
        return tree
    elif depth <=1:
        tree["value"] = np.mean(data_y)
        return tree
    else:
        data_index = np.arange(0, len(data_y), 1)  # 获取数据的编号，便于之后选取最佳分割点

        bestproperty,left_data_x,left_data_y,right_data_x,right_data_y=getBestProperty(data_x, data_y, data_index)

        tree["property"] = bestproperty[2]
        tree["t"] = bestproperty[1]

        tree["leftTree"] = creatTree(left_data_x,left_data_y, depth-1)
        tree["rightTree"] = creatTree(right_data_x,right_data_y, depth-1)

    return tree

# -----------------------------
#预测函数
def predict(test_x,tree):
    predict_result = []     #测试集通过决策树的预测结果
    for i in range(test_x.shape[1]):
        data = test_x[:,i]
        res = predict_digui(data, tree)
        predict_result.append(res)
    return predict_result

# -----------------------------
#预测使用的递归函数
def predict_digui(data,tree):

    if "property" in tree:
        property_x = data[tree["property"]]
        if property_x < tree["t"]:
            return(predict_digui(data,tree["leftTree"]))
        else:
            return(predict_digui(data, tree["rightTree"]))
    else:
        return tree["value"]

# -----------------------------
#模块测试
if __name__ == "__main__":
    #训练数据
    data_x = np.arange(0, 10, 0.2)
    data_x.shape = (1, data_x.shape[0])
    data_y = np.sin(data_x[0])+(np.random.rand(data_x.shape[1])/2-0.25)

    #开始训练
    tree = creatTree(data_x, data_y, 5)
    print("-----------------------------")
    print("训练出的决策树为：")
    print(tree)

    #测试数据
    test_x = np.arange(0, 10, 0.1)
    test_x.shape = (1, test_x.shape[0])
    #决策树开始预测
    test_y = predict(test_x,tree)
    print("测试数据的测试结果为：")
    print(test_y)
    print("-----------------------------")

    #绘制训练样本的散点图和测试样本的训练结果
    plt.plot(data_x[0],data_y,"r*",label="Train data scater")
    plt.plot(test_x[0],test_y,label="Test data line")
    plt.title("Decision Tree")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()