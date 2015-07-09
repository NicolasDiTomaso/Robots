import csv
from random import shuffle

fin = open('data_format2/train_format2.csv','r')
fout = open('train_format2_processed.csv','w')
writer = csv.writer(fout, delimiter=',')

# Copio header en nuevos datos
line = fin.readline()
writer.writerow(line.strip().split(',')[1:5])

# Preprocesamiento: se sacan registros con valores vacios o nulos y el activity_log y el user_id
repeatBuyers = []
nonRepeatBuyers = []
print "Starting"
for line in fin.readlines():
	data = [x for x in line.strip().split(',')[1:5] if x != '']
	# Se sacan los registros que tengan valores que esten marcados como desconocidos segun el formato de cada atributo
	if len(data) < 4 or data[0] == '0' or data[1] == '2' or data[3] == '-1':
		continue
	if data[3] == '1':
		repeatBuyers.append(data)
	elif data[3] == '0':
		nonRepeatBuyers.append(data);

# Tomo aleatoriamente la misma cantidad de registros que no se repitan
shuffle(nonRepeatBuyers)
nonRepeatBuyers = nonRepeatBuyers[:len(repeatBuyers)]

# Los ordeno aleatoriamente y se escriben en nuevo archivo de datos
toWrite = repeatBuyers + nonRepeatBuyers
shuffle(toWrite)
writer.writerows(toWrite)

print "Repeat: " + str(len(repeatBuyers)) + " No Repeat: " + str(len(nonRepeatBuyers))
print "Finished"