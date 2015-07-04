from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from numpy import genfromtxt
from itertools import islice
import csv

def readFromCsvIteration(file, iteration, step):
	with open(file, 'rb') as f:
		array = genfromtxt(islice(f, iteration*step ,iteration*step + step), delimiter=',',skip_header=1, usecols=range(0,5))
	return array

def activateNet():
	userid=int(raw_input('user id:'))
	agerange=int(raw_input('age range:'))
	gender=int(raw_input('gender:'))
	merchant_id=int(raw_input('merchant id:'))
	
	print net.activate([userid, agerange, gender, merchant_id])
	activateNet()

print "ITERATION 0"
array = readFromCsvIteration('/home/lara/FIUBA/ROBOTS/data_format2/train_format2.csv',0, 100)
number_of_columns = array.shape[1]

ds = SupervisedDataSet(number_of_columns - 1, 1)
ds.setField('input', array[:,:-1])
ds.setField('target', array[:,-1:])

net = buildNetwork(ds.indim,8,8,ds.outdim,recurrent=True)

t = BackpropTrainer(net,learningrate=0.001,momentum=0.05)
t.trainUntilConvergence(ds,maxEpochs=200)

for i in range(1, 10):
	print "ITERATION " + str(i)
	array = readFromCsvIteration('/home/lara/FIUBA/ROBOTS/data_format2/train_format2.csv',i, 100)
	number_of_columns = array.shape[1]

	ds = SupervisedDataSet(number_of_columns - 1, 1)
	ds.setField('input', array[:,:-1])
	ds.setField('target', array[:,-1:])

	t.trainUntilConvergence(ds,maxEpochs=200)

activateNet()