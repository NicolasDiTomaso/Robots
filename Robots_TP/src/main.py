from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import csv

ds = SupervisedDataSet(4, 1)

fin = open('train_format2_processed.csv','r')
fin.readline()

print "Reading data"
i = 0
testData = []
for line in fin.readlines():
	if i < 100000: # tomo una muestra de datos para el entrenamiento
		trainRow = [float(x) for x in line.strip().split(',')]
		ds.addSample(tuple(trainRow[:4]),tuple(trainRow[4:]))
		i = i + 1
	elif i < 200000: # tomo una muestra de datos para las pruebas
		testRow = [float(x) for x in line.strip().split(',')]
		testData.append(testRow) 
		i = i + 1
	else: 
		break

net = buildNetwork(ds.indim,8,8,ds.outdim,recurrent=True)
t = BackpropTrainer(net, learningrate=0.001,momentum=0.05,verbose=True)

print "Training"
t.trainUntilConvergence(ds, verbose=True, maxEpochs=10)

print "Testing"
fout = open('results.csv','w')
writer = csv.writer(fout, delimiter=',')
correct = 0
total = 0
for testRow in testData:
	predicted = round(net.activate(testRow[:4])[0])
	writer.writerow([testRow, predicted])
	if predicted == testRow[4]:
		correct = correct + 1
	total = total + 1
print "Precission: " + str(float(correct)/float(total))