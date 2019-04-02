import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D,axes3d

class GA:

    def __init__(self,variables_num=1,population_amount=10,chromosome_length=5,generations=100,copu_rate=0.5,mutation=0.03,domain=[[-10,10],[-10,-10]]):

        self.variables_num = variables_num  # 目标函数的自变量个数
        self.pop_amount = population_amount     #种群大小、个数
        self.chrom_length = chromosome_length   #染色体长度
        self.genes = generations    #世代次数
        self.copu_rate = copu_rate  #交配中染色体交换的位数比例
        self.mota_rate = mutation   #基因突变率
        self.domain = domain    # 定义域，求解范围

        #将染色体编码，染色体高低顺序为从左到右, 每个变量分开编码，每个变量的染色体长度都为chrom_length，且种群的变量数也都为都为variables_num
        self.chromosome = np.random.randint(low=0,high=2,size=(self.pop_amount,self.variables_num,self.chrom_length))

        pass

    # 目标函数
    def function(self,variables):
        X = variables[:,0]
        Y = variables[:,1]
        #X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)
        return Z
    # 适应度函数
    def suff_func(self,variables):
        result = self.function(variables)    # 若用竞标赛方法进行选择则不需要对适应度值进行处理
        # result[np.where(result<0)]=0  # 轮盘赌时需要将负数的适应度值进行处理（统一增加一个值或设定最小为0）
        return result

    # 染色体解码函数
    def encoding(self):
        index = np.arange(0,self.chrom_length)
        coef = np.exp2(index)
        decimal = np.sum(self.chromosome*coef,axis=2)   # 对每一个个体进行解码

        # 将解码的十进制值压缩到定义域内
        decimal_res = []
        for dec in range(decimal.shape[1]):
            res = self.domain[dec][0]+decimal[:,dec]*(self.domain[dec][1]-self.domain[dec][0])/(pow(2,self.chrom_length)-1)
            decimal_res.append(res)

        return np.array(decimal_res).T

    # 选择函数（通过竞标赛来选择）
    def select(self):
        # 解码得到十进制值
        decimal = self.encoding()
        # 计算适应度
        sufficiency = self.suff_func(decimal)
        chrom = []
        for i in range(self.pop_amount):
            spool = np.array(random.sample(range(self.pop_amount), int(self.pop_amount * 0.8)))  # 先随机产生一个选择池
            res = np.argmax(sufficiency[spool]) # 找出选择池中值最大的元素索引
            chrom.append(self.chromosome[spool[res]])   # 得到适应度最大的个体

        self.chromosome = np.array(chrom)
        pass

    # 选择函数（通过轮盘赌来选择）
    def select_RG(self):
        # 解码得到染色体的实际值
        decimal = self.encoding()

        # 求解每个值得适应度
        sufficiency = self.suff_func(decimal)
        # 求和
        suff_sum = np.sum(sufficiency)

        # 通过轮盘赌选取个体
        percentage = sufficiency/suff_sum   # 每个个体所占得比例
        cum_percentage = np.cumsum(percentage)  # 为比例求累加和
        P = np.random.random(self.pop_amount)   # 随机产生pop_amount个0-1随机数
        chrom = []
        for p in P:
            chrom.append(self.chromosome[np.where(cum_percentage>=p)][0])

        self.chromosome = np.array(chrom)
        pass

    # 交配函数、染色体交叉互换
    # 所有染色体均参与交叉配对
    def copulation(self):

        # res = self.chromosome[:,:,[0,1,2]]
        #
        # res[[0,1],0,:] = res[[1,0],0,:]
        #
        # self.chromosome[:,:,[0,1,2]] = res

        # 选取用于交配的染色体位数,和所在的位置
        chrom_col_index = np.array(random.sample(range(self.chrom_length),int(self.chrom_length*self.copu_rate)))
        chrom_col = self.chromosome[:,:,chrom_col_index]  # 可用于交换的染色体位

        # 染色体交叉配对
        for v in range(self.variables_num):
            for i in range(0,chrom_col.shape[0]-1,2):
                chrom_col[[i,i+1],v,:]=chrom_col[[i+1,i],v,:]

        self.chromosome[:,:,chrom_col_index]=chrom_col

        pass

    # 基因突变函数
    def mutation(self):
        p = np.random.random(self.pop_amount)   # 种群中每个染色体突变的概率
        # self.chromosome[np.where(p<self.mota_rate)]
        if len(p[p<self.mota_rate])!=0:
            col=np.random.randint(0,self.chrom_length)
            self.chromosome[p < self.mota_rate,:,col]=self.chromosome[p < self.mota_rate,:,col]^1     # 异或取反，既突变

        pass

    # 开始遗传进化
    def propagate(self):
        while self.genes!=0:
            self.select()
            self.copulation()
            self.mutation()
            self.genes = self.genes-1
        best_result = self.encoding()[np.argmax(self.suff_func(self.encoding()))]
        return best_result

if __name__ == "__main__":

    domain = [[-70,70],[-50,50]]
    ga = GA(variables_num=2,population_amount=5,chromosome_length=10,generations=500,domain=domain)
    variables = np.array([ga.propagate()])
    result = ga.function(variables)
    print(result)


    X = np.arange(-5, 5, 0.15)
    Y = np.arange(-5, 5, 0.15)
    #X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    print(np.max(Z))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    #
    # plt.show()