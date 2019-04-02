import numpy as np
import random
import matplotlib.pyplot as plt

class GA:

    def __init__(self,population_amount=10,chromosome_length=5,generations=100,copu_rate=0.5,mutation=0.03,domain=[-10,10]):

        self.pop_amount = population_amount     #种群大小、个数
        self.chrom_length = chromosome_length   #染色体长度
        self.genes = generations    #世代次数
        self.copu_rate = copu_rate  #交配中染色体交换的位数比例
        self.mota_rate = mutation   #基因突变率
        self.domain = domain    # 定义域，求解范围

        #将染色体编码，染色体高低顺序为从左到右
        self.chromosome = np.random.randint(low=0,high=2,size=(self.pop_amount,self.chrom_length))

        pass

    #目标函数
    def function(self,x):
        y = np.cos(x)*x
        return y
    #适应度函数
    def suff_func(self,x):
        y = self.function(x)
        y[np.where(y<0)]=0
        return y

    #染色体解码函数
    def encoding(self):
        index = np.arange(0,self.chrom_length)
        coef = np.exp2(index)
        decimal = np.sum(self.chromosome*coef,axis=1)

        #将解码的十进制值压缩到定义域内
        decimal = self.domain[0]+decimal*(self.domain[1]-self.domain[0])/(pow(2,self.chrom_length)-1)

        return decimal

    #选择函数
    def select(self):
        #解码得到染色体的实际值
        decimal = self.encoding()

        #求解每个值得适应度
        sufficiency = self.suff_func(decimal)
        #求和
        suff_sum = np.sum(sufficiency)

        #通过轮盘赌选取个体
        percentage = sufficiency/suff_sum   #每个个体所占得比例
        cum_percentage = np.cumsum(percentage)  #为比例求累加和
        P = np.random.random(self.pop_amount)   #随机产生pop_amount个0-1随机数
        chrom = []
        for p in P:
            chrom.append(self.chromosome[np.where(cum_percentage>=p)][0])

        self.chromosome = np.array(chrom)
        pass

    #交配函数、染色体交叉互换
    #所有染色体均参与交叉配对
    def copulation(self):
        #选取用于交配的染色体位数
        chrom_col_index = np.array(random.sample(range(self.chrom_length),int(self.chrom_length*self.copu_rate)))
        chrom_col = self.chromosome[:,chrom_col_index]  #可用于交换的染色体位

        #染色体交叉配对
        for i in range(0,chrom_col.shape[0]-1,2):
            chrom_col[[i,i+1],:]=chrom_col[[i+1,i],:]

        self.chromosome[:,chrom_col_index]=chrom_col
        pass

    #基因突变函数
    def mutation(self):
        p = np.random.random(self.pop_amount)   # 种群中每个染色体突变的概率
        # self.chromosome[np.where(p<self.mota_rate)]
        if len(np.where(p<0.5)[0])!=0:
            col=np.random.randint(0,self.chrom_length)
            self.chromosome[np.where(p < 0.5),col]=self.chromosome[np.where(p < 0.5),col]^1     # 异或取反，既突变

        pass

    #开始遗传进化
    def propagate(self):
        while self.genes!=0:
            self.select()
            self.copulation()
            self.mutation()
            self.genes = self.genes-1
        best_result = self.encoding()[np.argmax(self.suff_func(self.encoding()))]
        return best_result

if __name__ == "__main__":

    ga = GA(80,10,generations=500,domain=[-70,70])
    ga.encoding()
    res_x = ga.propagate()
    res_y = ga.function(res_x)

    x = np.arange(-70,70,1)
    y = ga.function(x)
    print(res_x,res_y)
    plt.plot(x,y,"b",label="Fuc graph")
    plt.scatter(res_x,res_y,marker="o",color="r",label="Best result")
    plt.legend()
    plt.show()