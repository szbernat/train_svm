#!/usr/bin/env python3

from sklearn import datasets
import csv

iris = datasets.load_iris()

header = iris["feature_names"]
header.append("class")
data_rows = []
for row,label in zip(iris["data"], iris["target"]):
    data_rows.append(list(row))
    data_rows[-1].append(label)

with open("iris.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_rows)
