#!/usr/bin/env python3

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import combinations

def make_meshgrid(x, y, h=.02):
    x_min, x_max = min(x) - 1, max(x) + 1
    y_min, y_max = min(y) - 1, max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

data = []
target = []
header = []
with open("iris.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader) # Skip data labels
    for row in reader:
        data.append(list(map(lambda x: float(x), row[:4])))
        target.append(int(row[4]))

svm_kernel = 'rbf'
comb = combinations(range(4), 2)

fig, axs = plt.subplots(2,3)
for c, ax in zip(comb, axs.flatten()):
    reduced_data = [[row[i] for i in c] for row in data]

    x_train,x_test,y_train,y_test = train_test_split(reduced_data, target, test_size=0.30, random_state=1997)

    svc = SVC(kernel=svm_kernel)
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)*100

    # Create plot
    x = [row[0] for row in x_test]
    y = [row[1] for row in x_test]
    xx, yy = make_meshgrid(x,y)

    z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x,y,c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlabel(header[c[0]])
    ax.set_ylabel(header[c[1]])
    ax.set_title(f"Accuracy = {accuracy:5.1f}%")
    fig.suptitle(f"SVMs with {svm_kernel} kernel", fontsize=24)

#  plt.show()
plt.tight_layout()
plt.savefig(f"{svm_kernel}.png")
