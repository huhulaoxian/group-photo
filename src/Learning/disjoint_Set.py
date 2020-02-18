# Python3 program to find 
# the number of islands using 
# Disjoint Set data structure. 

# Class to represent 
# Disjoint Set Data structure 

import logging

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='new.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
class DisjointUnionSets: 
	def __init__(self, n):
		self.rank = [0] * n 
		self.parent = [0] * n 
		self.n = n 
		self.makeSet() 

	def makeSet(self): 
		
		# Initially, all elements are in their 
		# own set. 
		for i in range(self.n): 
			self.parent[i] = i 

	# Finds the representative of the set that x 
	# is an element of 
	def find(self, x): 
		if (self.parent[x] != x): 

			# if x is not the parent of itself, 
			# then x is not the representative of 
			# its set. 
			# so we recursively call Find on its parent 
			# and move i's node directly under the 
			# representative of this set 
			return self.find(self.parent[x]) 
		return x 

	# Unites the set that includes x and 
	# the set that includes y 
	def Union(self, x, y): 
		
		# Find the representatives(or the root nodes) 
		# for x an y 
		xRoot = self.find(x) 
		yRoot = self.find(y) 

		# Elements are in the same set, 
		# no need to unite anything. 
		if xRoot == yRoot: 
			return

		# If x's rank is less than y's rank 
		# Then move x under y so that depth of tree 
		# remains less 
		if self.rank[xRoot] < self.rank[yRoot]: 
			self.parent[xRoot] = yRoot

		# Else if y's rank is less than x's rank 
		# Then move y under x so that depth of tree 
		# remains less 
		elif self.rank[yRoot] < self.rank[xRoot]: 
			self.parent[yRoot] = xRoot 

		else: 
			
			# Else if their ranks are the same 
			# Then move y under x (doesn't matter 
			# which one goes where) 
			self.parent[yRoot] = xRoot 

			# And increment the the result tree's 
			# rank by 1 
			self.rank[xRoot] = self.rank[xRoot] + 1

# Returns number of islands in a[][] 
def countIslands(a,cluser_num): 
	n = len(a) 
	m = len(a[0]) 
	if cluser_num != 0:
		for j in range(0, n): 
			for k in range(0, m):
				if a[j][k]!= cluser_num:
					a[j][k] = 0
				else :
					a[j][k] = 1
	else:
		for j in range(0, n): 
			for k in range(0, m):
				if a[j][k] == cluser_num:
					a[j][k] = 1
				else:
					a[j][k] = 0
	dus = DisjointUnionSets(n * m) 

	# The following loop checks for its neighbours 
	# and unites the indexes if both are 1. 
	for j in range(0, n): 
		for k in range(0, m): 

			# If cell is 0, nothing to do 
			if a[j][k] == 0: 
				continue


			# Check all 8 neighbours and do a Union 
			# with neighbour's set if neighbour is 
			# also 1 
			if j + 1 < n and a[j + 1][k] == 1: 
				dus.Union(j * (m) + k, 
						(j + 1) * (m) + k) 
			if j - 1 >= 0 and a[j - 1][k] == 1: 
				dus.Union(j * (m) + k, 
						(j - 1) * (m) + k) 
			if k + 1 < m and a[j][k + 1] == 1: 
				dus.Union(j * (m) + k, 
						(j) * (m) + k + 1) 
			if k - 1 >= 0 and a[j][k - 1] == 1: 
				dus.Union(j * (m) + k, 
						(j) * (m) + k - 1) 
			if (j + 1 < n and k + 1 < m and
					a[j + 1][k + 1] == 1): 
				dus.Union(j * (m) + k, (j + 1) *
							(m) + k + 1) 
			if (j + 1 < n and k - 1 >= 0 and
					a[j + 1][k - 1] == 1): 
				dus.Union(j * m + k, (j + 1) *
							(m) + k - 1) 
			if (j - 1 >= 0 and k + 1 < m and
					a[j - 1][k + 1] == 1): 
				dus.Union(j * m + k, (j - 1) *
							m + k + 1) 
			if (j - 1 >= 0 and k - 1 >= 0 and
					a[j - 1][k - 1] == 1): 
				dus.Union(j * m + k, (j - 1) *
							m + k - 1) 

	# Array to note down frequency of each set 
	c = [0] * (n * m) 
	numberOfIslands = 0
	lis = []
	list = []
	for j in range(n): 
		for k in range(m): 
			if a[j][k] == 1: 
				lis.append((j,k))
				x = dus.find(j * m + k) 
				
				# If frequency of set is 0, 
				# increment numberOfIslands 
				if c[x] == 0: 
					numberOfIslands += 1
					c[x] += 1
				else: 
					c[x] += 1

	list.append(numberOfIslands)
	list.append(lis)
	return list 



# Driver Code 
# kmeans_cluster_num = 4
# dic = {}
# for cluster_num in range(kmeans_cluster_num):
# 	g = [[1, 1, 0, 0, 0], 
# 		[0, 1, 0, 3, 3], 
# 		[1, 0, 0, 1, 1], 
# 		[0, 0, 3, 1, 2], 
# 		[1, 0, 1, 2, 0]] 
# 	# print("Number of ",cluster_num,"Islands is:", countIslands(g,cluster_num)) 
# 	dic[cluster_num] = countIslands(g,cluster_num)
# s = list(segments(dic))
# f24(1,s)
# f25(1,dic)



# kmeans_cluster_num = 12
# good_indices = list(np.load('../../data/good_indices.npy'))
# feature_vec = []
# LUV = np.load('../../data/LUV.npy')
# IH = np.load('../../data/IH.npy')
# for i, index in enumerate(good_indices):
# 	print ("图：",i+1)
# 	feature_vec.append([])
# 	dic ={}
# 	_LUV = LUV[i].reshape((16384, 3))
# 	for cluster_num in range(kmeans_cluster_num):
# 	# dic[cluster_num] = g.countIslands(cluster_num)
# 		kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0).fit(_LUV)
# 		centers = kmeans.cluster_centers_
# 		graph = kmeans.labels_ 
# 		graph = graph.reshape((128,128))
# 		# print("Number of ",cluster_num,"Islands is:", countIslands(graph,cluster_num)) 
# 		dic[cluster_num] = countIslands(graph,cluster_num)
# 	s = list(segments(dic))
# 	H = []
# 	for k in range(5):
# 		sumh = 0
# 		for i1, j1 in s[k]:
# 			sumh += IH[i][i1][j1]
# 		H.append(sumh)

# 	feature_vec[i].append(f26(i, s))
# 	feature_vec[i].append(f46(i, H))
# 	feature_vec[i].append(f47(i, H))

# This code is contributed by ankush_953 
