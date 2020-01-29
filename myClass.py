import csv
import numpy as np
from numpy import genfromtxt
import random
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def getMaxSwinglengthCSV(fileList):
	lens=[]
	nets =0
	for i in range(0, len(fileList)):		# go thru all swing files to get max length
		with open(fileList[i]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			Hr = list(csv.reader(csvDataFile))
			lens.append(len(Hr))
			Hr= genfromtxt(fileList[i], delimiter=',')
			data = Hr[:,0:3]
			x= Hr[0,4]
			# if tt==0:
				# print("this is a ZERO file!!",fileList[i])
	lll = np.array(lens)
	print("avg seq length mean is ",lll.mean(axis=0))
	return(max(lens))
	
def mapSwingsCategorical(fileList, ml, features):  # and filter nets 
	numFiles =len(fileList)
	output_array = np.zeros((numFiles,ml,features))  # train data is an array of (swingFiles, maxLength,features)
	bigArr = np.zeros((numFiles*ml,features))
	Xtar = []
	netCount = 0
	ii=0
	j = 0 
	lengths = []
	for i in range(0,len(fileList)):
		with open(fileList[i]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			Hr = list(csv.reader(csvDataFile))
		result = np.zeros((ml,features)) 					# place holder for variable length data
		ting = Hr[0][features+1:features+3]
		x = ting[0]
		y = ting[1]
		
	
		if x == "10" or x == "NaN":  		# 10 or 9 means net
			netCount=netCount+1
			continue
		else:
			Hr= genfromtxt(fileList[i], delimiter=',')
			data = Hr[:,0:features]
			lengths.append(len(data))
			result[:data.shape[0],:data.shape[1]] = data # put the data
			output_array[j,:,:] = result
			j = j+1
			bigArr[ii:ii+data.shape[0],:] = data
			ii = ii +data.shape[0]
			if x == '-1':
				Xtar.append(0)
				continue
			elif x == '-0.5':
				Xtar.append(1)
				continue
			elif x == '0':
				Xtar.append(2)
				continue
			elif x == '0.5':
				Xtar.append(3)
				continue
			elif x == '1':
				Xtar.append(4)
				continue
			else:
				print("WHOAA WHAT IS THIS!", x)
	
	Xtar = np.array(Xtar)
	goodSwings = numFiles-netCount
	bigArr = bigArr[:ii,:]
	print("bigArr shape in map categorical", bigArr.shape[0],"and ii is", ii)
	avg = bigArr.mean(axis=0)
	std = bigArr.std(axis=0)
	output_array=output_array[:goodSwings,:,:]
	print("output array shape mapSwingFunc ",output_array.shape)
	ass = checkZeros(output_array)
	if ass >0:
		print("bug is in map swings!")
	print("total length shouod be ",sum(lengths))
	return output_array, Xtar, bigArr
	
#def normFeature(fileList,mean,std,ml):	 ## go thru data, subtract each feature its mean and divide by  
def normFeature(x,mean,std,ml):
	features = x.shape[2]
	numFiles = x.shape[0]
	output_array = np.zeros((numFiles,ml,features))
	netCount = 0
	ii=0
	y_new = []
	bigArr = np.zeros((numFiles*ml,features))
	print("normalizing now!")
	for i in range(x.shape[0]):
		result = np.zeros((ml,features))
		seq =x[i,:,:]	# training sequence
		a, b = np.where(seq !=0.)
		lastNonZ = a[-1]+1
		seq2 =seq[:lastNonZ,:] # og training sequence with no zero fillers
		seq2 = seq2-mean  # subtract mean
		seq2 = seq2/std  # subtract mean
		bigArr[ii:ii+seq2.shape[0],:] = seq2
		ii = ii + seq2.shape[0]
		result[:seq2.shape[0],:seq2.shape[1]] = seq2 # put the data
		output_array[i,:,:] = result

	bigArr = bigArr[:ii,:]
	
	return output_array
	
	

def shuffleData(x, y):		# shuffles data, x = seqs, y =targets 
	goodSwings = y.shape[0]
	print("NOW shuffling!!")
	if x.shape[0] == y.shape[0]: 
		x_1 = np.zeros(x.shape)
		y_1 = np.zeros(y.shape)
		r = list(range(goodSwings))
		random.shuffle(r)
		for i in range(len(r)):
			x_1[i,:,:] = x[r[i],:,:]
			y_1[i] = y[r[i]]
			t = np.sum(x[i,:,:],axis=0)
			tt = np.sum(t)
		return x_1, y_1
		
def checkZeros(x):
	count = 0 
	for i in range(x.shape[0]):
		seq = x[i,:,:]
		t = np.sum(seq,axis=0)
		tt = np.sum(t)
		if tt == 0:
			print(i)
			count = count+1
	return count
	
def shuffle(x,y):
	r = list(range(x.shape[0])) #random list ranging up to number of seqs
	random.shuffle(r)
	x_1 = np.zeros(x.shape)
	y_1 = np.zeros(y.shape)
	for i in range(len(r)):
		x_1[i,:,:] = x[r[i],:,:]
		y_1[i] = y[r[i]]
	return x_1, y_1
	
def addNoise(y):
	l = y.shape[0]
	s = np.random.normal(0, 1, l)
	return y+s
	
def plotHisory(epochs,loss,val_loss):
	epochs = range(1,epochs+1)		  
	plt.figure()
	plt.plot(epochs,loss,'bo',label= 'Training acc')
	plt.plot(epochs,val_loss,'b',label= 'Validation acc')
	plt.title('Training and validiation Accuracy')
	plt.legend()
	plt.show()
	
def arrangeBatchesPerClass(x,y,batchsize):
	t =np.unique(y)
	ind_0, batch0, lo0 = batchClass(y,t[0],batchsize)
	ind_1, batch1, lo1 = batchClass(y,t[1],batchsize)
	ind_2, batch2, lo2 = batchClass(y,t[2],batchsize)
	ind_3, batch3, lo3 = batchClass(y,t[3],batchsize)
	ind_4, batch4, lo4 = batchClass(y,t[4],batchsize)	

	bbb=batch1+batch2+batch3+batch4+batch0
	r = list(range(bbb)) # is master batch counter, batch incrementer
	random.shuffle(r)
	c = y.shape[0]%batchsize
	c = y.shape[0]-c
	newTar = np.zeros((y[0:c].shape))
	newX = np.zeros((x[0:c,:,:].shape))
	
	# print("newTar shape", newTar.shape)
	# print("newX shape", newX.shape)
	k = 0
	
	for i in range(batch0):
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random position finish
		inds=ind_0[i*batchsize:i*batchsize+batchsize]
		newTar[here:there] = y[inds]
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1
	
	for i in range(batch1):  # for i random places 
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random +5
		inds = ind_1[i*batchsize:i*batchsize+batchsize]  # one batch indices corresponding to one target
		newTar[here:there] = y[inds]
		seqs = x[inds,:,:]
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1	
		
	for i in range(batch2):
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random position finish
		inds = ind_2[i*batchsize:i*batchsize+batchsize]
		newTar[here:there] = y[inds]
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1	
	
	for i in range(batch3):
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random position finish
		inds = ind_3[i*batchsize:i*batchsize+batchsize]
		newTar[here:there] = y[inds]
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1	
		
	for i in range(batch4):
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random position finish
		inds = ind_4[i*batchsize:i*batchsize+batchsize]
		newTar[here:there] = y[inds]
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1	
		
	a = checkZeros(newX)
	if a > 0:
		print('goodJob!')
		print("this is leftovers",lo2[0], lo4[0])
		lo1 = lo1.tolist()
		lo1.append(int(lo2[0]))
		lo1.append(int(lo4[0]))
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize 
		newTar[here:there] = y[lo1]
		newX[here:there,:,:] =x[lo1,:,:]
	
	if k >= len(r):
		coolbeans=0
		# print("ok. this is k and totalbatches", k, len(r))
		# print("new FUNC!")
	return newX, newTar
	

def batchClass(y, target, batchsize): #  return the indices locations and number of batches, multiple of batch size
	inds = np.where(y==target)
	inds= sum(inds)
	cl = inds.shape[0]
	c = cl%batchsize
	leftOver=[]
	if c >0:
		leftOver = inds[cl-c:]
		print("dropping",target,"samples",c, "indices",leftOver)
		inds = inds[0:cl-c]
		batches = inds.shape[0]/batchsize
		return inds, int(batches), leftOver  
	inds = inds[0:cl-c]
	batches = inds.shape[0]/batchsize
	return inds, int(batches),0 # return the indices locations and number of batches
	
def getTargetPairs(tar,x,y):
	print("target pairs")
	inds = np.where(y==tar)
	inds= sum(inds)
	seq = x[inds[0],:,:]
	print(seq)
	f1=seq[:,0]
	test = np.where(f1==0.)
	print(test)
	print("test type",type(test))
	print(test[0][0])
	tt = int(test[0][0])
	seq1 = seq[:tt,:]
	print("seq1 shape",seq1.shape)
	print(z)
	print("seq shape",seq.shape)
	noZ = np.where(seq != 0.)
	a, b = noZ
	seq1 = seq[:a.shape[0],:]
	print("seq1 shape",seq1.shape)
	print("a and b", a.shape, type(a), b.shape, type(b))
	print("##################################################")
	print("noZ",noZ)
	print("##################################################")
	print("seq no zeros ",seq[noZ])
	cc=seq[noZ] 
	print("seq no zeros shape ",cc.shape)
	print(z)
	xx = x[inds,:,:]
	return xx
	
def getSwings(fileList,ml,target):  # and filter nets 
		numFiles =len(fileList)
		output_array = np.zeros((numFiles,ml,3))  # train data is an array of (swingFiles, maxLength,features)
		bigArr = np.zeros((numFiles*ml,3))
		ii=0
		j = 0 
		lengths = []
		for i in range(0,len(fileList)):
			with open(fileList[i]) as csvDataFile:
				csvReader = csv.reader(csvDataFile)
				Hr = list(csv.reader(csvDataFile))
			result = np.zeros((ml,3)) 					# place holder for variable length data
			ting = Hr[0][4:6]
			x = ting[0]
			y = ting[1]
		
			if x == str(target):
				Hr= genfromtxt(fileList[i], delimiter=',')
				data = Hr[:,0:3]
				result[:data.shape[0],:data.shape[1]] = data # put the data
				output_array[j,:,:] = result
				bigArr[ii:ii+data.shape[0],:] = data
				ii = ii +data.shape[0]
				j = j+1
	
		return bigArr[:ii,:]
		
def splitBatches(x,y,split, batch):   # split data into batches based on float value 
	batchNum = y.shape[0]/batch
	print("number of total batches ",batchNum)
	tsplit = int(batchNum*split)
	dec = tsplit/batchNum
	ss= int((1-dec)*y.shape[0])
	x_t = x[:ss,:,:]
	y_t = y[:ss]
	x_val = x[ss:,:,:]
	y_val = y[ss:]
	return x_t, y_t, x_val, y_val
	
def getOneTarget(x,y,tar):
	inds = np.where(y==tar)
	inds = sum(inds)
	yy = y[inds]
	xx = x[inds,:,:]
	return xx, yy
		

def arrangeBatchesBinary(x,y,batchsize): # groups batches 
	t =np.unique(y)
	ind_1, batch1 = batchClass(y,t[0],batchsize)
	ind_2, batch2 = batchClass(y,t[1],batchsize)
	bbb=batch1+batch2
	print("bbb is ",bbb, "type", type(bbb))
	r = list(range(bbb)) # is master batch counter, batch incrementer
	random.shuffle(r)
	c = y.shape[0]%batchsize
	c = y.shape[0]-c
	newTar = np.zeros((y[0:c].shape))
	newX = np.zeros((x[0:c,:,:].shape))
	print("newTar shape", newTar.shape)
	print("newX shape", newX.shape)
	k = 0
	
	for i in range(batch1):  # for i random places 
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random +5
		inds = ind_1[i*batchsize:i*batchsize+batchsize]  # one batch indices corresponding to one target
		newTar[here:there] = y[inds]
		seqs = x[inds,:,:]
		print("seqs", seqs.shape)
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1	
		
	for i in range(batch2):
		j =r[k]
		here =j*batchsize  #random position start
		there = j*batchsize+batchsize #random position finish
		inds = ind_2[i*batchsize:i*batchsize+batchsize]
		newTar[here:there] = y[inds]
		newX[here:there,:,:] =x[inds,:,:]
		k=k+1	
	
		
	if k >= len(r):
		print("ok. this is k and totalbatches", k, len(r))
		print("new FUNC!")
	return newX, newTar
		
			

def printMatrix(predictions, targets,ii): # (savename,predicted,correct)print and save matrices
	target_names= ['0','1','2','3','4']
	print(classification_report(np.argmax(targets,axis=1), predictions, target_names=target_names))#, target_names=target_names))
	confM = confusion_matrix(np.argmax(targets,axis=1), predictions)
	print(confM)
	print(type(confM))
	plotTruePredict(np.argmax(targets,axis=1),predictions, ii)
	confM= np.array(confM)
	np.savetxt(ii+"confMatrix.csv",confM,delimiter=",")
	
def printMatrixSparse(predictions,targets,ii): # print and save matrices
	target_names= ['0','1','2','3','4']
	print(classification_report(targets, predictions,target_names=target_names))#, target_names=target_names))
	confM = confusion_matrix(targets, predictions)
	print(confM)
	confM= np.array(confM)
	np.savetxt(ii+"confMatrix.csv",confM,delimiter=",")
		
		
def getCategoricalData(features): # gets data, makes targets categorical 
	root = tkinter.Tk()
	root.withdraw()
	filez = filedialog.askopenfilenames(parent=root,title='SELECT FEATURE files')
	fileList = list(filez)
	print("total swing files", len(fileList))
	ml = getMaxSwinglengthCSV(fileList)
	#x,y1,bigArr = mc.mapSwingsCategorical(fileList, ml)  # returns seqs, targets, and long arr of features
	return mapSwingsCategorical(fileList,ml,features)

def mapToFloat(y):
	targets = np.unique(y)
	print(targets)
	print(y)
	inds = np.where(y==targets[0])
	print(inds)
	print(z)
	inds = sum(inds)
	y[inds] = -16
	inds = np.where(y==targets[1])
	inds = sum(inds)
	y[inds] = -8
	inds = np.where(y==targets[2])
	inds = sum(inds)
	y[inds] = 0
	inds = np.where(y==targets[3])
	inds = sum(inds)
	y[inds] = 8
	inds = np.where(y==targets[4])
	inds = sum(inds)
	y[inds] = 16
	
	y = addNoise(y)
	return 

	
def mapSwingsCategoricalMIN(fileList, ml, features):  # and filter nets 
	numFiles =len(fileList)
	output_array = np.zeros((numFiles,ml,features))  # train data is an array of (swingFiles, maxLength,features)
	Xtar = []
	netCount = 0
	ii=0
	j = 0 
	lengths = []
	allMins=[]
	bigArr = np.zeros((numFiles*ml,features))
	
	for i in range(0,len(fileList)):
		with open(fileList[i]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			Hr = list(csv.reader(csvDataFile))
			
		result = np.zeros((ml,features)) 					# place holder for variable length data
		ting = Hr[0][features+1:features+3]
		x = ting[0]
		y = ting[1]
	
		if x == "10" or x == "NaN":  		# 10 or 9 means net
			netCount=netCount+1
			continue
		else:
			min = Hr[0][features+3]
			allMins.append(int(min))
			Hr= genfromtxt(fileList[i], delimiter=',')
			data = Hr[:,0:features]
			lengths.append(len(data))
			result[:data.shape[0],:data.shape[1]] = data # put the data
			output_array[j,:,:] = result
			j = j+1
			
			bigArr[ii:ii+data.shape[0],:] = data
			ii = ii + data.shape[0]
			
			if x == '-1':
				Xtar.append(0)
				continue
			elif x == '-0.5':
				Xtar.append(1)
				continue
			elif x == '0':
				Xtar.append(2)
				continue
			elif x == '0.5':
				Xtar.append(3)
				continue
			elif x == '1':
				Xtar.append(4)
				continue
			else:
				print("WHOAA WHAT IS THIS!", x)
	Xtar = np.array(Xtar)
	allMins = np.array(allMins)
	goodSwings = numFiles-netCount
	output_array=output_array[:goodSwings,:,:]
	bigArr = bigArr[:ii,:]
	# std = bigArr.std(axis=0)
	# mean = bigArr.mean(axis=0)
	# print('bigarr mean and std is',mean)
	# print (std)
	print('sum lengths is',sum(lengths))
	print('ii lengths is',ii)
	print("output array shape mapSwingFunc ",output_array.shape)
	ass = checkZeros(output_array)
	if ass >0:
		print("bug is in map swings!")
	print("total length shouod be ",sum(lengths))
	return output_array, Xtar, allMins, bigArr
	
def shiftAlign(x, mins,fixedPt):
	shiftedX = np.zeros((x.shape[0],x.shape[1]+40,x.shape[2]))
	print('empty x array',x.shape)
	ii = 0
	for i in range(x.shape[0]):
		seq = x[i,:,:]
		min = mins[i]
		a, b = np.where(seq !=0.)
		lastNonZ = a[-1]+1
		#print("put something this long",lastNonZ)
		seq2 = seq[:lastNonZ,:]
		shift= fixedPt-min
		# print('shifted by this mSuch', shift)
		# print('into something this big', shiftedX[i,shift:shift+lastNonZ,:].shape)
		start = shift
		endd = shift+lastNonZ
		newSeq = np.zeros(seq.shape)
		shiftedX[i,shift:shift+lastNonZ,:] = seq2
		
	return shiftedX
	
def saveSparsePlots(history, eps,name):
	epochs= range(1,eps+1)
	plt.figure()
	plt.plot(epochs, history.history['loss'])
	plt.plot(epochs, history.history['val_loss'])
	plt.legend(['Training','Validation'])
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.savefig(name+'LOSS.png')
	loss= []
	loss.append(history.history['loss'])
	loss.append(history.history['val_loss'])
	loss = np.array(loss)
	np.savetxt(name+"LOSS.csv",loss,delimiter=",")
	
	plt.figure()
	plt.plot(epochs, history.history['sparse_categorical_accuracy'])
	plt.plot(epochs, history.history['val_sparse_categorical_accuracy'])
	plt.legend(['Training','Validation'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.savefig(name+'ACCURACY.png')
	acc= []
	acc.append(history.history['sparse_categorical_accuracy'])
	acc.append(history.history['val_sparse_categorical_accuracy'])
	acc = np.array(acc)
	np.savetxt(name+"ACC.csv",acc,delimiter=",")
	
	plt.figure()
	plt.plot(epochs, history.history['mean_absolute_error'])
	plt.plot(epochs, history.history['val_mean_absolute_error'])
	plt.legend(['Training','Validation'])
	plt.xlabel('Epochs')
	plt.ylabel('MAE')
	plt.savefig(name+'MAerror.png')
	
	
def saveScatter(predicted, actual,name):
	print(type(predicted))
	print(predicted)
	print(predicted.shape)
	yo=predicted.shape[0]
	yo =int(yo)
	yo=range(1,yo+1)
	plt.figure()
	plt.scatter(yo,predicted,color='r',marker ='x')
	plt.scatter(yo,actual,color='b')
	plt.legend(['Predicted','Actual'])
	plt.savefig(name+'Scatter.png')
	
def plotTruePredict(true, predicted,name):
	fig = plt.figure()
	plt.plot(true,color='b',linestyle=':',marker='.')	
	plt.plot(predicted,color='r',linestyle=':',marker='.')
	plt.legend(['True','Predicted'])
	plt.title('Target Comparison')
	plt.ylim([-1,5])
	plt.ylabel('target')
	plt.xlabel('swing')
	fig.tight_layout()
	plt.savefig(name+'ValTargets.png')
	c = np.absolute(true-predicted) 
	print('absolute error for ', name,c,c.mean(axis=0))
	
	
def savePlots(history, eps,name):
	epochs= range(1,eps+1)
	plt.figure()
	plt.plot(epochs, history.history['loss'])
	plt.plot(epochs, history.history['val_loss'])
	plt.legend(['Training','Validation'])
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.savefig(name+'LOSS.png')
	loss= []
	loss.append(history.history['loss'])
	loss.append(history.history['val_loss'])
	loss = np.array(loss)
	np.savetxt(name+"LOSS.csv",loss,delimiter=",")
	
	plt.figure()
	plt.plot(epochs, history.history['categorical_accuracy'])
	plt.plot(epochs, history.history['val_categorical_accuracy'])
	plt.legend(['Training','Validation'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.savefig(name+'ACCURACY.png')
	acc= []
	acc.append(history.history['categorical_accuracy'])
	acc.append(history.history['val_categorical_accuracy'])
	acc = np.array(acc)
	np.savetxt(name+"ACC.csv",acc,delimiter=",")
	
	plt.figure()
	plt.plot(epochs, history.history['mean_absolute_error'])
	plt.plot(epochs, history.history['val_mean_absolute_error'])
	plt.legend(['Training','Validation'])
	plt.xlabel('Epochs')
	plt.ylabel('MAE')
	plt.savefig(name+'MAerror.png')
	
	
	
	
		






