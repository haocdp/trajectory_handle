import csv

file = csv.reader(open("K:/毕业论文/TaxiData_Porto/train.csv", 'r'))

for line in file:
    print(line)
