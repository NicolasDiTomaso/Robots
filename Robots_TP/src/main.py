#Agregar modulo pybrain a la carpeta src o incluirla al pythonpath

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

#creo una red generica
net = buildNetwork(2, 3, 1)

#pruebo de activar un valor para ver que funcione
print (net.activate([2, 1]))

#Creo un dataset de 2 elementos de entrada y 1 de salida
ds = SupervisedDataSet(2, 1)

#le cargo algunos valores(XOR):
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

#Vemos un poco los datos
print len(ds)

for inpt, target in ds:
    print inpt, target
    
print ds['input']

print ds['target']

net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)

print trainer.train()

print trainer.trainUntilConvergence()