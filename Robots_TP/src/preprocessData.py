import csv
from random import shuffle

fin = open('data_format2/train_format2.csv','r')
fout = open('train_format2_processed.csv','w')
writer = csv.writer(fout, delimiter=',')

# manteniendo header
line = fin.readline()
writer.writerow(line.strip().split(',')[1:5])

length = 50000

# Preprocesamiento: se sacan registros con valores vacios o nulos y el activity_log y el user_id
repeatBuyers = []
nonRepeatBuyers = []
print "Starting"
for line in fin.readlines():
	data = [x for x in line.strip().split(',')[1:5] if x != '']
	if len(data) < 4 or data[0] == '0' or data[1] == '2' or data[3] == '-1': # removing unknown data
		continue
	if data[3] == '1' and len(repeatBuyers) < length: # 5000 de repetidos
		repeatBuyers.append(data)
	if data[3] == '0' and len(nonRepeatBuyers) < length: # 5000 de no repetidos
		nonRepeatBuyers.append(data);
	
	if len(repeatBuyers) >= length and len(nonRepeatBuyers) >= length:
		break

toWrite = repeatBuyers + nonRepeatBuyers
shuffle(toWrite)
writer.writerows(toWrite)

print "Finished"