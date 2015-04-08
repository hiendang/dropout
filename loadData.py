import pandas as pd
from sklearn import preprocessing
import numpy as np

np.random.seed(1988);
print ("Loading loadData module and seed numpy random with 1988");

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
	
def loadTestSet():
	test = pd.read_csv('test.csv');
	test = test.drop('id', axis=1);
	return test;

def saveResult(preds,filename):
	sample = pd.read_csv('sampleSubmission_o.csv')
	preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
	preds.to_csv(filename, index_label='id')
	return;