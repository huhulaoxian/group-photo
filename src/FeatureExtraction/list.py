import numpy as np
dic = {}
dic1 = {}
dic[0]=[1, 2]
dic1[0]=['1','2']
a = [1,2,3,7,5,6,7,20,9,10]

print(a[-5:][::-1])
print(dic[0][0])
print(dic1[0][0])
b = [1,2]
s1,s2 =b
print("s1=",s1,type(s1),"s2=",s2)
b+='3'
print(1-2**-(1))

y = ['ads',0,0,0,1,1,1,1]
for index,i in enumerate(y):
    if i == 1:
        print(index)
a = [1,2,3,7,5,6,7,20,9,10]
a = a[:3]
print(a)
print(np.load("rf_face_ranking.npy")+1)
print(np.sort(np.load("rf_ranking.npy")[:25]+1))
print(np.sort(np.load("rf_without_ranking.npy")[:20]+8))
