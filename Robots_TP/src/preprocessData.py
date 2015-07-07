import csv

fin = open('/home/lara/FIUBA/ROBOTS/data_format2/train_format2.csv','r')
fout = open('train_format2_processed2.csv','w')
writer = csv.writer(fout, delimiter=',')

# Preprocesamiento: se sacan registros con valores vacios o nulos y el activity_log y el user_id
print "Starting"
for line in fin.readlines():
	data = [x for x in line.strip().split(',')[1:5] if x != '']
	if len(data) < 4:
		continue
	writer.writerow(data)
print "Finished"