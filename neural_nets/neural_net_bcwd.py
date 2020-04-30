import pandas as pd 
import numpy as np
from random import shuffle


def read_file():
    data_set = pd.read_csv('/home/thedarkcoder/Desktop/ML/Coursera/datasets/breast-cancer-wisconsin.data.csv')
    file = open('/home/thedarkcoder/Desktop/ML/Coursera/datasets/general_file.txt','w')
    file.write(str(data_set['label'].value_counts()))
    file.close()
    return data_set

def classify_sets():
    list1 = []
    file = open('/home/thedarkcoder/Desktop/ML/Coursera/datasets/general_file.txt','r')
    for i in file:
        l = i.split()
        list1.append(l[0])
    list1.remove('Name:')
    file.close()
    return list(map(float, list1))

def train_test_split(data):
    data_set = data.values.astype("float64").tolist()
    shuffle(data_set)
    train_data = data_set[:int(0.8 * len(data))]
    test_data = data_set[int(0.8 * len(data)):]
    return train_data, test_data

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feedforward(a0,w0,b0,w1,b1):
    a1 = sigmoid(np.dot(w0,a0) + b0)
    out = sigmoid(np.dot(w1,a1) + b1)
    return a1, out

def cost_compute(cost_arr, out):
    cost = 0
    cost_arr = cost_arr - out
    for i in range(cost_arr.shape[0]):
        cost += cost_arr[i, 0]**2
    return 0.5*cost, cost_arr

def backprop(cost_arr, out, a1, w1, a0):
    dw1 = np.array([0]*a1.shape[0]).reshape((a1.shape[1],a1.shape[0]))
    db1 = np.zeros(out.shape)
    dw0 = np.array([0]*a0.shape[0]).reshape((a0.shape[1],a0.shape[0]))
    db0 = np.zeros(a1.shape)

    for i in range(cost_arr.shape[0]):
        dw_1 = cost_arr[i, 0] * out[i, 0] * (1 - out[i, 0])  * np.reshape(a1,(a1.shape[1], a1.shape[0]))
        dw1 = np.concatenate((dw1, dw_1), axis = 0)
    dw1 = np.delete(dw1, 0, 0)

    for i in range(cost_arr.shape[0]):
        db1[i, 0] = cost_arr[i, 0] * out[i, 0] * (1 - out[i, 0])

    dw = 0; dw_0 = 0
    for i in range(a1.shape[0]):
        for j in range(out.shape[0]):
            dw += cost_arr[j, 0] * out[j, 0] * (1 - out[j, 0]) * w1[j, i] 
        dw_0 = dw * a1[i, 0] * (1 - a1[i, 0]) * np.reshape(a0,(a0.shape[1], a0.shape[0]))
        dw0 = np.concatenate((dw0, dw_0), axis = 0)
    dw0 = np.delete(dw0, 0, 0)

    db = 0
    for i in range(a1.shape[0]):
        for j in range(out.shape[0]):
            db += cost_arr[j, 0] * out[j, 0] * (1 - out[j, 0]) * w1[j, i] 
        db0[i, 0] = db * a1[i, 0] * (1 - a1[i, 0])
    
    return dw1, db1, dw0, db0


data = read_file()
data.drop('id', 1, inplace=True)
data.drop('x6', 1, inplace=True)
data.drop('x9', 1, inplace=True)
ans_list = classify_sets()
train_list, test_list = train_test_split(data)
# print(backprop(np.array([1,0]).reshape((2,1)),np.array([0.981,0.021]).reshape((2,1)), np.array([0.997,0.001,0.756, 0.211]).reshape((4,1)),np.zeros((4,7)),np.zeros((7,1))))

cost_list = [[1.0, 0.0], [0.0, 1.0]]
w0 = np.zeros((4, len(train_list[0][:-1])))
w1 = np.zeros((2, 4))
b0 = np.zeros((4,1))
b1 = np.zeros((2,1))

cost = 0; cost_arr = None
for i in train_list[:]:
    a0 = np.asarray(i[:-1]).reshape((7,1))
    a1, out = feedforward(a0,w0,b0,w1,b1)
    if i[-1] == 2:
        cost, cost_arr = cost_compute(np.asarray(cost_list[0]).reshape((2,1)),out)
    else:
        cost, cost_arr = cost_compute(np.asarray(cost_list[1]).reshape((2,1)),out)
    dw1, db1, dw0, db0 = backprop(cost_arr, out, a1, w1, a0)
    w1 += 0.5 * dw1; w0 += 0.1 * dw0; b1 += 0.1 * db1; b0 += db0
    # w1 += dw1; w0 += dw0; b1 += db1; b0 += db0

count = 0
for i in test_list:
    a0 = np.asarray(i[:-1]).reshape((7,1))
    a1,out = feedforward(a0, w0, b0, w1, b1)
    pred = max(out)
    # print(out,"\t",i[-1])
    k, j = np.where(np.isclose(out, pred))
    if k == 0:
        print('2',i[-1])
    if k == 1:
        print('4',i[-1])
    if (k == 0 and i[-1] == 2.0) or (k == 1 and i[-1] == 4.0):
        count += 1 

print(count, len(test_list))
acc = count/len(test_list)
print(acc)

