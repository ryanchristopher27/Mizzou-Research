import csv

file = open('data/test_data.csv')

type(file)

csvreader = csv.reader(file)

header = []
header = next(csvreader)

print(header)

rows = []
for row in csvreader:
    rows.append(row)

for row in rows:
    print(row)

file.close()
