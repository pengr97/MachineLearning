"""demo.py"""
import numpy as np
import geatpy as ga  # 导入geatpy库
import matplotlib.pyplot as plt
import time

"""============================目标函数============================"""


def aim(x):  # 传入种群染色体矩阵解码后的基因表现型矩阵
    return x * np.sin(10 * np.pi * x) + 2.0


x = np.linspace(-1, 2, 200)
plt.plot(x, aim(x))  # 绘制目标函数图像
start_time = time.time()  # 开始计时
"""============================变量设置============================"""
x1 = [-1, 2]  # 自变量范围
b1 = [1, 1]  # 自变量边界
codes = [1]  # 变量的编码方式，2个变量均使用格雷编码
precisions = [5]  # 变量的精度
scales = [0]  # 采用算术刻度
ranges = np.vstack([x1]).T  # 生成自变量的范围矩阵
borders = np.vstack([b1]).T  # 生成自变量的边界矩阵
"""========================遗传算法参数设置========================="""
NIND = 40;  # 种群个体数目
MAXGEN = 25;  # 最大遗传代数
GGAP = 0.9;  # 代沟：说明子代与父代的重复率为0.1
"""=========================开始遗传算法进化========================"""
FieldD = ga.crtfld(ranges, borders, precisions, codes, scales)  # 调用函数创建区域描述器
Lind = np.sum(FieldD[0, :])  # 计算编码后的染色体长度
Chrom = ga.crtbp(NIND, Lind)  # 根据区域描述器生成二进制种群
variable = ga.bs2rv(Chrom, FieldD)  # 对初始种群进行解码
ObjV = aim(variable)  # 计算初始种群个体的目标函数值
pop_trace = (np.zeros((MAXGEN, 2)) * np.nan)  # 定义进化记录器，初始值为nan
ind_trace = (np.zeros((MAXGEN, Lind)) * np.nan)  # 定义种群最优个体记录器，记录每一代最优个体的染色体，初始值为nan
# 开始进化！！
for gen in range(MAXGEN):
    FitnV = ga.ranking(-ObjV)  # 根据目标函数大小分配适应度值(由于遵循目标最小化约定，因此最大化问题要对目标函数值乘上-1)
    SelCh = ga.selecting('sus', Chrom, FitnV, GGAP)  # 选择，采用'sus'随机抽样选择
    SelCh = ga.recombin('xovsp', SelCh, 0.7)  # 重组(采用单点交叉方式，交叉概率为0.7)
    SelCh = ga.mutbin(SelCh)  # 二进制种群变异
    variable = ga.bs2rv(SelCh, FieldD)  # 对育种种群进行解码(二进制转十进制)
    ObjVSel = aim(variable)  # 求育种个体的目标函数值
    [Chrom, ObjV] = ga.reins(Chrom, SelCh, 1, 1, 1, -ObjV, -ObjVSel, ObjV, ObjVSel)  # 重插入得到新一代种群
    # 记录
    best_ind = np.argmax(ObjV)  # 计算当代最优个体的序号
    pop_trace[gen, 0] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    pop_trace[gen, 1] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
    ind_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的变量值
# 进化完成
end_time = time.time()  # 结束计时
"""============================输出结果及绘图================================"""
print('目标函数最大值：', np.max(pop_trace[:, 0]))  # 输出目标函数最大值
variable = ga.bs2rv(ind_trace, FieldD)  # 解码得到表现型
print('用时：', end_time - start_time)
plt.plot(variable, aim(variable), 'bo')

plt.show()
