import xlrd
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------
#读取源数据
source_datas = xlrd.open_workbook('W3.0αDateSet.xlsx')
table = source_datas.sheet_by_index(0)
#获取变量值
X = []  #保存所有影响因素值
for i in range(1,table.nrows):
    row_val = table.row_values(rowx=i,start_colx=1,end_colx=3)
    row_val.append(1)
    X.append(row_val)

#获取函数值，结果
col_value = table.col_slice(colx=3,start_rowx=1,end_rowx=None)
Y = [data.value for data in col_value]
for i in range(len(Y)):     #将结果转化为1和0的型式
    if Y[i]=='是' :
        Y[i] = 1
    else :
        Y[i]=0

#--------------------------------------------------------
#最大似然估计估计参数，使用牛顿迭代求最优解
#牛顿迭代函数
def NtIteration(accuracy,S_X,S_Y):

    beta = np.array([0, 0, 1])  # 初始化beta
    l_val= 0   #保存的上一次函数值

    while True:
        lBetaRes = np.dot(beta,S_X)     #beta和测试集的矩阵乘法,结果为一个矩阵

        l_val_new = 0  # 最小化函数值
        for i in range(len(S_X[0])):
            l_val_new+=np.log(1+np.exp(lBetaRes[i]))-S_Y[i]*lBetaRes[i]

        #当差值在误差范围内时，达到条件，退出迭代
        if(np.abs(l_val_new-l_val)<=accuracy):
            break

        l_val = l_val_new   #保存上一个最小化函数值

        betaG1 = 0  #beta的一阶导数
        betaG2 = 0  #beta的二阶导数
        #求beta的一阶导数和二阶导数
        for i in range(len(S_X[0])):
            betaG1 = betaG1 - np.dot(np.array([S_X[:,i]]).T,(S_Y[i]-1.0/(1+np.exp(-lBetaRes[i]))))
            betaG2 = betaG2 + np.dot(np.array([S_X[:,i]]).T,np.array([S_X[:,i]]))*1.0/(1+np.exp(-lBetaRes[i])*(1-1.0/(1+np.exp(-lBetaRes[i]))))

        beta.shape = (3,1)
        beta = beta - np.dot(np.linalg.inv(betaG2),betaG1)  #beta的迭代公式
        beta = beta.T[0]
    return beta
# --------------------------------------------------------
# 模型验证函数，同时将验证结果排序便于求解P、R、TPR、FPR
def ModelVerification(T_X,T_Y,beta):
    T_result = []  # 用于保存验证得到的Y值
    lBetaRes = np.dot(beta, T_X)
    for i in range(len(T_X[0])):
        y = 1.0 / (1 + np.exp(-lBetaRes[i]))  # 目标函数
        T_result.append(y)  # 将验证结果加入T_result数组
        if i > 0:
            j = i
            while T_result[j] > T_result[j - 1] and j > 0:  # 将验证得到的结果按照从大到小的顺序排列，用于绘制P-R曲线
                temp = [T_result[j], T_Y[j]]
                T_result[j] = T_result[j - 1]
                T_result[j - 1] = temp[0]
                T_Y[j] = T_Y[j - 1]
                T_Y[j - 1] = temp[1]
                j -= 1
    return T_result

# --------------------------------------------------------
#求TPR,FPR函数
def ROCcurve(T_result,T_Y):
    # ROC曲线
    TPR = [0]
    FPR = [0]
    for i in range(len(T_result)):
        if T_Y[i] == 1:
            TPR.append(TPR[i] + 1.0 / 8)
            FPR.append(FPR[i])
        elif T_Y[i] == 0:
            FPR.append(FPR[i] + 1.0 / 9)
            TPR.append(TPR[i])
    return TPR,FPR
#--------------------------------------------------------
#模型训练
accuracy = 0.00001  # 牛顿迭代的精度值

#将整个数据集作为训练集
S_X = np.array(X).T  # 训练集自变量
S_Y = np.array(Y)    # 训练集因变量

#调用牛顿迭代函数进行模型训练
beta = NtIteration(accuracy,S_X,S_Y)    #参数矩阵

# --------------------------------------------------------
# 模型验证
T_X = np.array(X).T  # 由于数据集太小，所以将整个数据集用作模型评估的样本
T_Y = np.array(Y)
T_result = ModelVerification(T_X, T_Y, beta)  # 调用模型验证函数进行模型验证
TPR_ori, FPR_ori = ROCcurve(T_result, T_Y)  # 获取真正例率，假正例率
beta_ori = beta
AUC_ori = 0     # 通过计算每次训练出的模型所产生的ROC曲线面积AUC，最终选取面积最大的最优模型
for i in range(1, len(TPR_ori) - 1):
    AUC_ori += 0.5 * (FPR_ori[i + 1] - FPR_ori[i]) * (TPR_ori[i] + TPR_ori[i + 1])
# 错误率与精度
error_rate_ori = 0
for i in range(len(T_result)):
    if round(T_result[i]) != T_Y[i]:
        error_rate_ori += 1.0 / len(T_result)

# --------------------------------------------------------
# 参数调节获取最优模型
AUC = []    #ROC曲线下的面积
step = 0.05
count = 10
parameter1 = beta[0]
parameter2 = beta[1]
parameter3 = beta[2]

beta_new = []   #保存所有参数组合
error_rate = [] #保存所有错误率
for i in range(count):
    for j in range(count):
        for k in range(count):
            p1 = parameter1-(count/2)*step+i*step
            p2 = parameter2-(count/2)*step+j*step
            p3 = parameter3-(count/2)*step+k*step
            beta_new.append([p1,p2,p3])
            T_X1 = np.array(X).T
            T_Y1 = np.array(Y)
            T_result = ModelVerification(T_X1, T_Y1, np.array([p1,p2,p3]))  # 调用模型验证函数进行模型验证
            TPR, FPR = ROCcurve(T_result, T_Y1)  # 获取真正例率，假正例率
            AUC_new = 0     # 通过计算每次训练出的模型所产生的ROC曲线面积AUC，最终选取面积最大的最优模型
            for m in range(1, len(TPR) - 1):
                AUC_new += 0.5 * (FPR[m + 1] - FPR[m]) * (TPR[m] + TPR[m + 1])
            AUC.append(AUC_new)
            #获取每种参数组合下的错误率
            er = 0
            for i in range(len(T_result)):
                if round(T_result[i]) != T_Y[i]:
                    er += 1.0 / len(T_result)
            error_rate.append(er)

# --------------------------------------------------------
# 通过模型的AUC值获取最优模型
T_X = np.array(X).T
T_Y = np.array(Y)
AUC = np.array(AUC)
beta = np.array(beta_new)
error_rate = np.array(error_rate)
beta = beta[np.where(AUC==np.max(AUC))]  #获取模型优化后使得AUC最大的beta值
error_rate = error_rate[np.where(AUC==np.max(AUC))]     #获取使得错误率最小的
beta = beta[np.where(error_rate==np.min(error_rate))][0]    #再通过错误率删选使得错误率最小的

T_result = ModelVerification(T_X,T_Y,beta)
TPR,FPR = ROCcurve(T_result,T_Y)        #获取最终TPR,FPR用于绘制ROC图

# --------------------------------------------------------
# 模型评估，包括错误率、查准率、查全率、P-R曲线、ROC曲线
# 错误率与精度
error_rate = 0
for i in range(len(T_result)):
    if round(T_result[i]) != T_Y[i]:
        error_rate += 1.0 / len(T_result)

# 查准率，查全率，P-R曲线
TP = 0  # 真正例
FN = 0  # 假反例
FP = 0  # 假正例
TN = 0  # 真反例
for i in range(len(T_result)):
    if round(T_result[i]) == T_Y[i] and T_Y[i] == 1:
        TP += 1
    elif round(T_result[i]) == T_Y[i] and T_Y[i] == 0:
        TN += 1
    elif round(T_result[i]) != T_Y[i] and round(T_result[i]) == 1:
        FP += 1
    elif round(T_result[i]) != T_Y[i] and round(T_result[i]) == 0:
        FN += 1

#通过插补法，插入均值，防止因为测试集较少出现的分母为零的情况
P = TP / (TP + FP) if (TP + FP)!=0 else 0.5  # 查准率
R = TP / (TP + FN) if (TP + FN)!=0 else 0.5  # 查全率

#获取P-R曲线上坐标点
PR_Y = []   #P-R曲线的纵坐标值
PR_X = []   #P-R曲线的横坐标值
for i in range(0, len(T_result)):
    TP = 0  # 真正例
    FN = 0  # 假反例
    FP = 0  # 假正例
    TN = 0  # 真反例
    for j in range(0, len(T_result)):
        if T_Y[j] == 1 and j <= i:
            TP += 1
        elif T_Y[j] == 0 and j <= i:
            FP += 1
        elif T_Y[j] == 1 and j > i:
            FN += 1
        elif T_Y[j] == 0 and j > i:
            TN += 1
    P1 = TP / (TP + FP) if (TP + FP) != 0 else 0.5  # 查准率
    R1 = TP / (TP + FN) if (TP + FN) != 0 else 0.5  # 查全率
    PR_Y.append(P1)
    PR_X.append(R1)
PR_Y = np.array(PR_Y)
PR_X = np.array(PR_X)

# --------------------------------------------------------
#输出模型以及度量结果
print("-------------------------------------")
print("优化前的模型：")
print("beta值：",beta_ori)
print("错误率：",error_rate_ori,"精度为：",1-error_rate_ori)
print("AUC值为：",AUC_ori)
print("优化后的模型：")
print("beta值：",beta)
print("错误率：",error_rate,"精度为：",1-error_rate)
print("查准率P：",P,"查全率R：",R)
print("AUC值为：",np.max(AUC))
print("-------------------------------------")

# --------------------------------------------------------
#绘制最优模型的P-R图，ROC图
#原始图像
plt.figure()
plot0 = plt.plot(FPR_ori,TPR_ori, label='ROC_ori')
plot1 = plt.plot(FPR,TPR, label='ROC')
plot2 = plt.plot(PR_X, PR_Y, label='P-R')
plt.xlabel("R/FPR")
plt.ylabel("P/TPR")
plt.title("P-R Curve/ROC Line")
plt.legend()

#经过拟合后的图像
plt.figure()
z1 = np.polyfit(FPR, TPR, 3)#用3次多项式拟合
p1 = np.poly1d(z1)
yvals=p1(FPR)#也可以使用yvals=np.polyval(z1,x)
z2 = np.polyfit(FPR_ori, TPR_ori, 3)
p2 = np.poly1d(z2)
yvals2=p2(FPR_ori)#也可以使用yvals=np.polyval(z1,x)
z3 = np.polyfit(PR_X, PR_Y, 2)
p3 = np.poly1d(z3)
yvals3=p3(PR_X)#也可以使用yvals=np.polyval(z1,x)

plot0 = plt.plot(FPR_ori,yvals2, label='ROC_ori')
plot1 = plt.plot(FPR,yvals, label='ROC')
plot2 = plt.plot(PR_X, yvals3, label='P-R')
plt.xlabel("R/FPR")
plt.ylabel("P/TPR")
plt.title("P-R Curve/ROC Curve")
plt.legend()
plt.show()
