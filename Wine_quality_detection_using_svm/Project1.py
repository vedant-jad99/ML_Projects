#Here we are training model to classify the wine quality in two labels namely good and bad using 
# Support Vector Machine from scikit learn
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn import svm


#Here we are again classifying the wine in two classes, "good" and "bad". The wine with quality above
#6 (7 and 8) are classified as good and those with 6 and below are classifies as "bad". The revised dataset is returned
def new_characterise(data):

    #Converting data to a dictionary. All the columns in dataFrame become elements of dictionary.
    data = dict(data)
    n = len(data['quality'])
    
    #The values held by each dictionary is a numpy array. That is converted to a list for replacement.
    data['quality'] = data['quality'].tolist()

    #Here we are replacing each data entry with either good or bad.
    for i in range(n):
        if data['quality'][i] < 7:
            data['quality'][i] = "bad"
        else:
            data['quality'][i] = "good"  

    #Again converting it to a dataframe.
    data = pd.DataFrame(data)
    return data


#Here data is split into training and testing data.
def split(X, y):    
    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    #read the csv file and fill out the blank spaces with 0
    data = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/winequality-red.csv")
    data.fillna(0, inplace= True)    
    
    data = new_characterise(data)

    #feature selection here. We are selecting seven features from the dataset
    X = data[['fixed acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'pH', 'alcohol']]
    y = data[['quality']]
    y = np.array(y).reshape(len(y), )


    X_train, X_test, y_train, y_test = split(X, y)

    #Here we are using SVM classifier with rbf kernel for classification.
    svc = svm.SVC()
    svc.fit(X_train, y_train)
    yhat = svc.predict(X_test)
    y_test = np.array(y_test)

    #Calculating the accuracy here.
    count = 0
    for i in range(len(yhat)):
        if yhat[i] == y_test[i]:
            count += 1
    print(count/len(y_test))
