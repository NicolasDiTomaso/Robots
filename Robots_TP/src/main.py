from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
import csv
import numpy as np

ds = SupervisedDataSet(3, 1)

fin = open('train_format2_processed.csv','r')
num_lines = sum(1 for line in open('train_format2_processed.csv','r'))
fin.readline()

length = (num_lines - 1) / 2

print "Reading data"
i = 0
testData = []
for line in fin.readlines():
	if i < length: # tomo una muestra de datos para el entrenamiento
		trainRow = [float(x) for x in line.strip().split(',')]
		ds.addSample(tuple(trainRow[:3]),tuple(trainRow[3:]))
		i = i + 1
	elif i < length * 2: # tomo una muestra de datos para las pruebas
		testRow = [float(x) for x in line.strip().split(',')]
		testData.append(testRow) 
		i = i + 1
	else: 
		break

# Normalizo los atributos
i = np.array([d[0] for d in ds])
i /= np.max(np.abs(i),axis=0)
o = np.array([d[1] for d in ds])
o /= np.max(np.abs(o),axis=0)

test = np.array([row for row in testData])
test /= np.max(np.abs(test), axis=0)

nds = SupervisedDataSet(3, 1)
for ix in range(len(ds)):
    nds.addSample( i[ix], o[ix])

net = buildNetwork(nds.indim,3,2,nds.outdim, recurrent=True)
t = BackpropTrainer(net,verbose=True)

print "Training"
t.trainUntilConvergence(nds,verbose=True, maxEpochs=30)

print "Testing"
fout = open('results.csv','w')
writer = csv.writer(fout, delimiter=',')
correct = 0
total = 0
for testRow in test:
	predicted = net.activate(testRow[:3])[0]
	roundedPredicted = round(predicted)
	writer.writerow(np.append(testRow, [roundedPredicted,predicted]))
	if roundedPredicted == testRow[3]:
		correct = correct + 1
	total = total + 1
print "Precission: " + str(float(correct)/float(total))