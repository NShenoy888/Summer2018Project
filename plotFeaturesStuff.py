import myClassNew as mc 
import tkinter
from tkinter import filedialog
import os
import numpy as np
from numpy import genfromtxt
import random
import matplotlib.pyplot as plt



if __name__ == "__main__":
###################################### training data ######
	
	root = tkinter.Tk()
	root.withdraw()
	filez = filedialog.askopenfilenames(parent=root,title='SELECT FEATURE files')
	fileList = list(filez)
	features = 7
	ml=111
	x,y,mins = mc.mapSwingsCategoricalMIN(fileList, ml,features)  # returns seqs, targets, and long arr of features
	print(x.shape, y.shape, mins.shape)

	
	Xa= mc.shiftAlign(x,mins,51)
	
	
	x0,y0 = mc.getOneTarget(Xa,y,0)
	x4,y4 = mc.getOneTarget(Xa,y,4)
	
	xx0 = np.transpose(x0[:,20:80,0])
	xx4 = np.transpose(x4[:,20:80,0])
	maxL = range(1,xx4.shape[1]+1)
	print(xx4.shape)
	maxL = np.linspace(0,2,xx4.shape[0])

	fig = plt.figure()
	fig.suptitle(' ')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylabel('degrees')
	plt.title('Target 0')
	plt.subplot(212)
	plt.title('Target 4')
	plt.ylabel('degrees')
	plt.xlabel('seconds')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	fig.tight_layout()
	plt.savefig('elbowAnglecompareFeats1.png')
	
	
	xx0 = np.transpose(x0[:,20:80,1])
	xx4 = np.transpose(x4[:,20:80,1])
	fig = plt.figure()
	fig.suptitle(' ')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylabel('degrees')
	plt.title('target 0')
	plt.subplot(212)
	plt.title('target 4')
	plt.ylabel('degrees')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	plt.xlabel('seconds')
	fig.tight_layout()
	plt.savefig('shoulderAnglecompareFeats2.png')
	
	xx0 = np.transpose(x0[:,20:80,2])
	xx4 = np.transpose(x4[:,20:80,2])
	fig = plt.figure()
	fig.suptitle(' ')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylabel('degrees')
	plt.title('target 0')
	plt.ylim([0,150])
	plt.subplot(212)
	plt.title('target 4')
	plt.ylim([0,150])
	plt.ylabel('degrees')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	plt.xlabel('seconds')
	fig.tight_layout()
	plt.savefig('armpitcompareFeats3.png')
	
	xx0 = np.transpose(x0[:,20:80,3])
	xx4 = np.transpose(x4[:,20:80,3])
	fig = plt.figure()
	fig.suptitle(' ')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylabel('cm')
	plt.title('target 0')
	plt.subplot(212)
	plt.title('target 4')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	plt.ylabel('cm')
	plt.xlabel('seconds')
	fig.tight_layout()
	plt.savefig('wristSHoulderDistcompareFeats4.png')
	
	xx0 = np.transpose(x0[:,20:80,4])
	xx4 = np.transpose(x4[:,20:80,4])
	fig = plt.figure()
	fig.suptitle(' ')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylim([0,10])
	plt.ylabel('cm')
	plt.title('target 0')
	plt.subplot(212)
	plt.title('target 4')
	plt.ylim([0,10])
	plt.ylabel('cm')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	plt.xlabel('seconds')
	fig.tight_layout()
	plt.savefig('shouderDisplacementcompareFeats5.png')
	
	
	xx0 = np.transpose(x0[:,20:80,5])
	xx4 = np.transpose(x4[:,20:80,5])
	fig = plt.figure()
	fig.suptitle(' ')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylim([0,160])
	plt.ylabel('Degrees')
	plt.title('target 0')
	plt.subplot(212)
	plt.title('target 4')
	plt.ylim([0,160])
	plt.ylabel('Degrees')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	plt.xlabel('seconds')
	fig.tight_layout()
	plt.savefig('hipRotatointCompare.png')
	
	
	xx0 = np.transpose(x0[:,20:80,6])
	xx4 = np.transpose(x4[:,20:80,6])
	fig = plt.figure()
	fig.suptitle('')
	plt.subplot(211)
	plt.plot(maxL,xx0,color='b',linestyle=':')
	plt.ylim([0,180])
	plt.ylabel('degrees')
	plt.title('target 0')
	plt.subplot(212)
	plt.title('target 4')
	plt.ylim([0,180])
	plt.ylabel('degrees')
	plt.plot(maxL,xx4,color='r',linestyle=':')
	plt.xlabel('seconds')
	fig.tight_layout()
	plt.savefig('facingAnglecompareFeats6.png')
	

	
	
	
	
	