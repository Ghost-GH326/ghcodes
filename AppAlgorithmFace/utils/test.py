import csv
 
with open('/home/guohao826/AppAlgorithmFace/material/c189ced2-dbb1-4d52-9a91-654bac488116-1.csv', 'r', errors='ignore') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row[0])
