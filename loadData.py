import pandas as pd
from sklearn import preprocessing
import numpy as np
from numpy import isnan
import cPickle
np.random.seed(1988);
print ("Loading loadData module and seed numpy random with 1988");

file = open('customFeature.pkl','r')
featurePairs = cPickle.load(file)
file.close()

def loadTrainSet():
	train = pd.read_csv('train.csv');
	
	labels = train.target.values;
	train = train.drop('id', axis=1);
	train = train.drop('target', axis=1);
	
	lbl_enc = preprocessing.LabelEncoder();
	labels = lbl_enc.fit_transform(labels);
	
	return (train,labels)

def loadShuffledTrain():
	(train,labels) = loadTrainSet();
	per = np.random.permutation(train.index)
	train.insert(0,'labels',labels)
	train = train.reindex(per);
	shuffledLabels = train.get('labels').values
	shuffledTrain = train.drop('labels',axis=1)
	return (shuffledTrain,shuffledLabels);

def splitTrainSet(trainData,trainLabels,train_ratio=0.8):
	msk = np.random.randn(len(trainData)) <train_ratio;
	subTrainData = trainData[msk];
	subValidData = trainData[~msk];
	subTrainLabels = trainLabels[msk];
	subValidLabels = trainLabels[~msk];
	return (subTrainData,subTrainLabels,subValidData,subValidLabels);

def preprocessingData(data):
	meanf = data.T.mean()
	varf = data.T.var()
	maxf = data.T.max()
	
	countNonZeros = data[data>0].count(axis=1)
	countOnes = data[data==1].count(axis=1)
	mean_without_Zeros = data[data>0].mean(axis=1)
	var_without_Zeros = data[data>0].var(axis=1)
	var_without_Zeros[isnan(var_without_Zeros)]=0;
	features = np.column_stack((meanf,varf,mean_without_Zeros,var_without_Zeros,maxf,countNonZeros,countOnes,data));
	d2 = data.values**2
	for (col1,col2) in featurePairs:		
		col = np.sqrt(d2[:,col1]+d2[:,col2])
		features = np.column_stack((col,features));
	return np.log(features+1);#np.log(features+1);
	
def loadTestSet():
	test = pd.read_csv('test.csv');
	test = test.drop('id', axis=1);
	return test;

def saveResult(preds,filename):
	sample = pd.read_csv('sampleSubmission_o.csv')
	preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
	preds.to_csv(filename, index_label='id')
	return;