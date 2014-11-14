#! /usr/bin/python
"""
Multiple Recursive-Patitioned Linear Regression
"""
import math
import numpy as np
from numpy import array
import pylab as pl
import sys
import copy
from sklearn.linear_model import LinearRegression
#import numpy.array as array
rawData = open("rawLogNormalized.csv",'r')
datas = rawData.readlines()
for i in range(len(datas)):
	datas[i] = datas[i].replace('\n','')
	datas[i] = datas[i].split(';')
	#print datas[i]
	del datas[i][5]
	del datas[i][3]
	del datas[i][0]
#PARAMS
ReliableRatio = 5
BRE_BOUNDARY_MRE = 0.05
BRE_BOUNDARY_MER = 0.05
#Hapus nama kolom
print datas[0]
del datas[0]
def euclidian(data1,data2):
	distance = 0
	for i in range(len(data1)):
		if data1[i] != '' and data2[i]!='':
			distance += math.pow(float(data1[i])-float(data2[i]),2)
	return math.sqrt(distance)

distance = range(len(datas))
for i in range(len(datas)):
	distance[i] = [0]*len(datas)
for i in range(len(datas)):
	for j in range(len(datas)):
		if i==j : continue
		if distance[i][j]==0:
			distance[i][j] = euclidian(datas[i],datas[j])
			distance[j][i] = distance[i][j]
def kNearestNeighbourFillMissing(data,neighbourDistance,k):
	global datas
	nearest = [999]*k
	nearestIndexes = [0]*k
	pp = zip(range(len(neighbourDistance)),neighbourDistance)
	mm = sorted(pp, key=lambda project: project[1])
	nearestNeighbour = mm[1:1+k]
	for i in range(len(data)):
		if data[i]=='':
			totalAttr = 0
			for j in nearestNeighbour:
					try:
						if datas[j[0]][i]=='':
								continue
						totalAttr+=float(datas[j[0]][i])
					except ValueError:
						print i
						print datas[j[0]]
						
			data[i] = totalAttr/k
	#print data

#isi nilai variabel yang hilang
for i in range(len(datas)):
	kNearestNeighbourFillMissing(datas[i],distance[i],2)
#buat model regresi
def getModelOutput(param,inputTraining):
	output = 0
	for i in range(len(inputTraining)):
		output += float(inputTraining[i])*param[i]
	return output
def renewParam(actualOutput,targetOutput,trainingInput,param,learningRate):
	for i in range(len(param)):
		sigmaResidu = 0
		for j in range(len(actualOutput)):
			try:
				sigmaResidu+=(actualOutput[j]-targetOutput[j])*float(trainingInput[j][i])
			except TypeError:
				#print trainingInput[j][i]
				sys.exit(0)
		sigmaResidu = float(sigmaResidu)/len(actualOutput)
		param[i] = param[i]-learningRate*sigmaResidu
def squareError(target,actual):
	totalError = 0
	for i in range(len(target)):
		totalError+=(target[i]-actual[i])**2
	return totalError
def buildRegressionModel(training,learningRate,initialParam):
	output = [0]*len(training)
	trainingInput = [0]*len(training)
	trainingOutput = [0]*len(training)
	for i in range(len(training)):
		trainingInput[i] = training[i][0:len(training[i])-1]
		trainingInput[i].insert(0,1) #Masukan Bias
		trainingOutput[i] = float(training[i][len(training[i])-1])
		output[i] = getModelOutput(initialParam,trainingInput[i])
	#print squareError(trainingOutput,output)
	lastSquareError = 0
	for i in range(7000):
		renewParam(output,trainingOutput,trainingInput,initialParam,learningRate)
		for j in range(len(training)):
			output[j] = getModelOutput(initialParam,trainingInput[j])
		#print squareError(trainingOutput,output)
		curSquareError = squareError(trainingOutput,output)
		if curSquareError==lastSquareError:
			break
		lastSquareError = curSquareError
	return squareError(trainingOutput,output)
class RegressionNode:
	def __init__(self,model,datas):
		self.model = model
		self.datas = datas
		self.up = None
		self.down = None
	def recursiveGetModel(self,needle):
		if self.datas.__contains__(needle):
			return self.model
		if self.up!=None:
			oo = self.up.recursiveGetModel(needle)
			if oo != False:
				return oo
		if self.down !=None:
				pp = self.down.recursiveGetModel(needle)
				if pp != False:
					return pp		
		return False
	def setUpperRegressionNode(self,node):
		self.up = node
	def getUpperRegressionNode(self):
		return self.up
	def setLowerRegressionNode(self,node):
		self.down = node
	def getLowerRegressionNode(self):
		return self.down
	def printModel(self):
		print self.model
		if self.up!=None:
			self.up.printModel()
		if self.down!=None:
			self.down.printModel()
	def printData(self):
		print self.datas
		if self.up!=None:
			self.up.printData()
		if self.down!=None:
			self.down.printData()
def BRE(estimated,actual):
	return abs(actual-estimated)/min([actual,estimated])
def partitionData(datas,initParam):
	if len(datas)<15:
		return False
	MRE1 = LinearRegression()
	#print type(datas)
	training_input = datas[:,[0,1,2]]
	training_output = datas[:,[3]]
	#print datas
	#print training_input
	#print training_output
	MRE1.fit(training_input,training_output)
	MSE = 0.0
	for i in range(len(training_input)):
		MSE += (training_output[i]-MRE1.predict(training_input[i]))**2
	MSE = MSE/len(training_input)
	cooksDistance = [0]*len(datas)
	chi_square_awal = 0.0
	#print "PARAMS"
	#print MRE1.coef_
	
	for i in range(len(datas)):
		masukan = datas[i][0:len(datas[i])-1]
		#masukan.insert(0,1)
		#print masukan
		#print MRE1.coef_
		#print MRE1.intercept_
		
		keluaran = MRE1.predict(masukan)
		chi_square_awal += ((keluaran-float(datas[i][3]))**2)/keluaran
	chi_square = [0]*len(datas)
	MRES = []
	for i in range(len(datas)):
		mData = copy.copy(datas)
		mData = np.delete(mData,i,0)
		mInitParam = [0.1,0.1,0.1,0.1]
		new_training_input = mData[:,[0,1,2]]
		new_training_output = mData[:,[3]]
		MRE = LinearRegression()
		MRE.fit(new_training_input,new_training_output)
		MRES.append(MRE)
		#ambil nilai sigma 
		sigma = 0
		new_chi_square = 0
		for j in range(len(mData)):
			masukan = mData[j][0:3]
			#print masukan
			#masukan.insert(0,1)
			actualObservation = float(mData[j][3])
			oldOutput = MRE1.predict(masukan)
			newOutput = MRE.predict(masukan) 
			sigma += (oldOutput-newOutput)**2
			new_chi_square += ((actualObservation-newOutput)**2)/actualObservation
		cooksDistance[i] = sigma/((len(mData[0])-1)*MSE)
		chi_square[i] = new_chi_square
	discarded = []
	for j in range(len(cooksDistance)):
		if cooksDistance[j] > (float(3*4)/len(cooksDistance)):
			discarded.insert(len(discarded),j)
		elif cooksDistance[j] > float(4)/len(cooksDistance):
			if chi_square[j]<chi_square_awal:
				discarded.insert(len(discarded),j)
	#BUAT MODEL BARU TANPA DATA BERMASALAH
	mData = copy.copy(datas)
	discarded.sort(reverse=True)
	for i in discarded:
		mData = np.delete(mData,i,0)
	newInitParam = [0.1,0.1,0.1,0.1]
	#MRE2 = buildRegressionModel(mData,0.1,newInitParam)
	MRE2 = LinearRegression()
	MRE2.fit(mData[:,[0,1,2]],mData[:,[3]])
	#print MRE2
	#selesai buat model baru
	
	overEstimated = []
	underEstimated = []
	totalIndex = []
	for i in range(len(mData)):
		actual = mData[i][len(mData[i])-1]
		#print mData[i][0:len(mData[i])-1]
		#print MRE2.coef_
		calculated = MRE2.predict(mData[i][0:len(mData[i])-1])#getModelOutput(newInitParam,mData[i][0:len(mData)-1])
		bre = BRE(calculated,float(actual))
	 	#print bre
	 	#print "%f %f %f"%(actual,calculated,bre)
	 	if actual>calculated and bre > BRE_BOUNDARY_MER:
			underEstimated.insert(len(underEstimated),mData[i])
			totalIndex.append(i)
		elif actual<calculated and bre > BRE_BOUNDARY_MER: 
			overEstimated.insert(len(overEstimated),mData[i])
			totalIndex.append(i)
	totalIndex.reverse()
	underEstimated = np.array(underEstimated)
	overEstimated = np.array(overEstimated)
	#print "---"
	#keluarkan yang overestimated dan yang underestimated
	for i in range(len(totalIndex)):
		#print underEstimatedIndex[i]
		#print mData
		mData = np.delete(mData,totalIndex[i],0)
	#for i in range(len(overEstimatedIndex)):
	#	mData = np.delete(mData,overEstimatedIndex[i],0)
	curRegressionNode = RegressionNode(MRE2,mData)
	lowerModel = [0.1,0.1,0.1,0.1]
	#print overEstimated
	lowerRegressionNode = partitionData(overEstimated,lowerModel)
	if lowerRegressionNode==False and len(overEstimated)>0:
		mData = np.append(mData,overEstimated,0)
	elif lowerRegressionNode!=False:
		curRegressionNode.setLowerRegressionNode(lowerRegressionNode)
	upperModel = [0.1,0.1,0.1,0.1]
	#print len(underEstimated)
	upperRegressionNode = partitionData(underEstimated,upperModel)
	if upperRegressionNode ==False and len(overEstimated)>0:
		mData = np.append(mData,underEstimated,0)
	elif upperRegressionNode!=False:
		curRegressionNode.setLowerRegressionNode(upperRegressionNode)
	#print mData
	curRegressionNode.datas = mData
	#refitting model
	#buildRegressionModel(mData,0.1,curRegressionNode.model)
	return curRegressionNode 	
initParam = [0.1,0.1,0.1,0.1]
#print datas[0]
"""
testItem = datas[0]
copyData = copy.copy(datas)
ff = copyData[0] 
del copyData[0]
p = partitionData(copyData,initParam)
#print initParam
p.printModel()

print p.datas
if p.up!=None:
	print p.up.datas
if p.down != None:
	print p.down.datas
print datas[19]
"""
#print p.recursiveGetModel(datas[19])
datas = array(datas,dtype='float')
"""
Build model using leave-one-out test
"""
for i in range(len(datas)):
	
	copyData = copy.copy(datas)
	pp = zip(range(len(distance[i])),distance[i])
	mm = sorted(pp, key=lambda project: project[1])
	nearestSignificantIndex = 2
	while nearestSignificantIndex<len(mm):
		if abs(mm[nearestSignificantIndex][1]-mm[1][1])<0.1:
			nearestSignificantIndex+=1
			continue
		else:
			break
	nearestSignificant = mm[1:nearestSignificantIndex]
	t = copyData[i]
	#del copyData[i]
	copyData = np.delete(copyData,i,0)
	mainPartition = partitionData(copyData,initParam)
	validModel = []
	for j in range(len(nearestSignificant)):
		y = mainPartition.recursiveGetModel(copyData[j])
		if y!=False:
			validModel.append(y)
		else:
			print "gagal Nemu"
	inputA = t[0:3]
	#inputA.insert(0,1)	
	estimated = 0
	for i in range(len(validModel)):
		estimated += validModel[i].predict(inputA)
	#print estimated
	estimated = float(estimated)/len(validModel)
	#print i
	print "%f;%f"%(estimated,float(t[3]))
	
#datas = array(datas)
#print datas[0,::3]
