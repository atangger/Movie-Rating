import numpy as np
import scipy.io as sio
import scipy.linalg as slag
import matplotlib.pyplot as plt
import pandas as pd 
from numpy import mat
def getOriginalIdx(idx,dict):
	for (k,v) in moviedic.items():
		if v == idx :
			return k

def getGradU(i,R,U,V,k):
	u_i = U[i,:]
	movnums = V.shape[0]
	gradU = np.zeros([1,k])
	for j in range(movnums):
		if R[i,j] != 0:
			v_j = V[j,:]
			# print("u_i = ", u_i)
			# print("v_j = ", v_j)
			# print("uvdot = ", np.dot(u_i,v_j))
			gradU += np.dot((R[i,j] - np.dot(u_i,v_j)),v_j)
	gradU *= -2
	return gradU / np.linalg.norm(gradU)

def getGradV(j,R,U,V,k):
	v_j = V[j,:]
	usernum = U.shape[0]
	gradV = np.zeros([1,k])

	testflag = 0
	for i in range(usernum):
		if R[i,j] != 0:
			u_i = U[i,:]
			gradV += np.dot((R[i,j] - np.dot(u_i,v_j.T)),u_i)
			testflag = 1

	return gradV / np.linalg.norm(gradV)

def lsqU(i,R,U,V,k,pl):
	u_i = U[i,:]
	movnums = V.shape[0]
	d = 0
	param_lambda = pl
	for j in range(movnums):
		if R[i,j] !=0:
			d+=1
	y = mat(np.zeros([d,1]))
	X = mat(np.zeros([d,k]))
	cnt = 0
	for j in range(movnums):
		if R[i,j] !=0:
			y[cnt,0] = R[i,j]
			X[cnt,:] = V[j,:]
			cnt+=1
	regul = param_lambda * mat(np.eye(k))
	res = (X.T*X + regul).I*X.T*y
	res = res.reshape([1,k])
	# print("res = ", res.reshape([1,50]))
	U[i,:] = res

def lsqV(j,R,U,V,k,pl):
	v_j = V[j,:]
	usrnums = U.shape[0]
	d = 0
	param_lambda = pl
	for i in range(usrnums):
		if R[i,j] !=0:
			d+=1
	y = mat(np.zeros([d,1]))
	X = mat(np.zeros([d,k]))
	cnt = 0
	for i in range(usrnums):
		if R[i,j] !=0:
			y[cnt,0] = R[i,j]
			X[cnt,:] = U[i,:]
			cnt+=1

	regul = param_lambda * mat(np.eye(k))
	res = (X.T*X + regul).I*X.T*y
	res = res.reshape([1,k])
	# print("res = ", res.reshape([1,50]))
	V[j,:] = res

def getLoss(R,U,V):
	res = 0
	for i in range(U.shape[0]):
		for j in range(V.shape[0]):
			if R[i,j] != 0:
				u_i = U[i,:]
				v_j = V[j,:]
				res += (R[i,j]- np.dot(u_i,v_j))**2
	return res

def getTestLoss(testSet,U,V):
	res = 0;
	for e in testSet:
		u_i = U[e[0],:]
		v_j = V[e[1],:]
		res += (e[2] - np.dot(u_i,v_j))**2
	return res

def getLossforUi(i,R,U,V):
	res = 0
	u_i = U[i,:]
	for j in range(V.shape[0]):
		if R[i,j] != 0:
			v_j = V[j,:]
			res += (R[i,j]- np.dot(u_i,v_j))**2
	return res

data=(pd.read_csv("movie_ratings.csv").values)
usernum = 0
movienum = 0
userdic = {}
moviedic = {}


for i in range(data.shape[0]):
	if not int(data[i][0]) in userdic:
		userdic[int(data[i][0])] = usernum
		usernum+=1
	if not int(data[i][1]) in moviedic:
		moviedic[int(data[i][1])] = movienum
		movienum+=1

R = np.zeros([usernum,movienum])

for i in range(data.shape[0]):
	usridxt = int(data[i][0])
	movidxt = int(data[i][1])
	R[userdic[usridxt],moviedic[movidxt]] = data[i][2]

testSetNum = 300
testSet = []
testSetCnt = 0
while testSetCnt < testSetNum:
	didx = np.random.randint(0,data.shape[0])
	i_r = userdic[int(data[didx][0])]
	j_r = moviedic[int(data[didx][1])]
	if R[i_r,j_r] == 0:
		continue
	# In case for making some usr or movie have no rating
	uCnt = 0 
	mCnt = 0
	for j in range(movienum):
		if R[i_r,j] != 0:
			uCnt+=1
	for i in range(usernum):
		if R[i,j_r] !=0:
			mCnt+=1
	if mCnt == 1 or uCnt ==  1:
		continue
	tmplst = [i_r,j_r,R[i_r,j_r]]
	# print("R[test] = ", R[i_r,j_r])
	testSet.append(tmplst)
	testSetCnt+=1
	R[i_r,j_r] = 0

param_k = 40
param_lambda = 0.5
param_lrate = 0.01
U = np.random.random([usernum,param_k])
V = np.random.random([movienum,param_k])

# Use Least Square
print("start Pre-Train")
param_pretrainNum = 2
for i in range(param_pretrainNum):
	print("bef train testloss = ",getTestLoss(testSet,U,V))
	for i in range(usernum):
		lsqU(i,R,U,V,param_k,param_lambda)
	for j in range(movienum):
		lsqV(j,R,U,V,param_k,param_lambda)
	print("after train testloss = ",getTestLoss(testSet,U,V))
print("start Training")

# Use Gradient Decent 
param_iteNum = 500

for ite in range(param_iteNum):
	print("bef train testloss = ",getTestLoss(testSet,U,V))
	print("bef train trainloss = ",getLoss(R,U,V))
	for i in range(usernum):
		gradU = getGradU(i,R,U,V,param_k)
		# print("gradU = ", gradU.reshape(-1))
		gradU = gradU.reshape(-1)
		U[i,:] += - param_lrate*gradU


	for j in range(movienum):
		gradV = getGradV(j,R,U,V,param_k)
		gradV = gradV.reshape(-1)
		V[j,:] += -param_lrate*gradV

	print("after train testloss = ",getTestLoss(testSet,U,V))
	print("after train trainloss = ",getLoss(R,U,V))