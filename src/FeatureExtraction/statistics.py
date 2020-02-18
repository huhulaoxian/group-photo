import csv
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab

# reader = csv.reader(open("../../data/image.csv"))
reader = csv.reader(open("../../data/score_decimal.csv"))
# y = [0]*10
# for row in reader:
#     i = int(row[2]) - 1 
#     y[i] = y[i] + 1
# print('(1-10分)评分分布：',y)

data = []
# reader = csv.reader(open("../../data/image.csv"))
reader = csv.reader(open("../../data/score_decimal.csv"))
for row1 in reader:
    data.append(float(row1[2]))

data1 = np.array(data)
print('平均分',data1.mean())
# plt.hist(data,bins=30)
mu = data1.mean()
sigma = data1.std()
n, bins, patches = plt.hist(data, 30, density=1, facecolor='green', edgecolor = 'black',alpha=0.5)  
#直方图函数，x为x轴的值，density=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象  
y = mlab.normpdf(bins, mu, sigma)#画一条逼近的曲线  
plt.plot(bins, y, 'r--') 
# plt.bar(range(0,10), y)
plt.xlabel('score')
plt.ylabel('probability')
plt.title(r'Histogram of Grade')#中文标题 u'xxx' 
plt.show()

# np.save('../../data/statistics.npy', y)