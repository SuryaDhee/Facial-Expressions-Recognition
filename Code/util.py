import numpy as np
import pandas as pd

def init_weight_and_bias(M1,M2):
    W = np.random.randn(M1,M2)/np.sqrt(M1+M2)
    b = np.zeros(M2)
    return W, b

def relu(x):
    return x * (x>0)

def sigmoid(A):
    return 1/(1+np.exp(-A))

def tanh(A):
    return np.tanh(A)

def softmax(A):
    expA = np.exp(A)
    return expA/expA.sum(axis=1,keepdims=True)

def sigmoid_cost(T,Y):
    return -(T*np.log(Y)+(1-T)*np.log(1-Y)).sum()

def cost(T,Y):
    return -(T*np.log(Y)).sum()

#Same as cost but more efficient
def cost2(T,Y):
    N = len(T)
    return -np.log(Y[np.arange(N),T]).sum() #Without zeros ; Indexing Y using T
    
    

def error_rate(targets,predictions):
    return np.mean(targets!=predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N,K))
    for i in range(N):
        ind[i,y[i]] = 1
    return ind

def getData(balance_ones=True):

    path = '../Data/face_data.csv'
    data = pd.read_csv(path)
    y = data['emotion']
    X = []
    for i in range(len(data)):
        X.append(data[' pixels'][i].split(' '))


    X = np.array(X)
    X = X.astype('float64')
    X = np.divide(X,255.0)
    y = np.array(y)


    if(balance_ones):
        X0, y0 = X[y!=1], y[y!=1]
        X1 = X[y==1]
        X1 = np.repeat(X1,len(X0)/len(X1),axis=0)
        X = np.vstack([X0,X1])
        y = np.concatenate((y0,[1]*len(X1)))

    return X,y

def getBinaryData(balance_ones=True):
    path = '../Data/face_data.csv'
    data = pd.read_csv(path)

    y = []
    X = []

    for i in range(len(data)):
        if(data.loc[i,'emotion']==0 or data.loc[i,'emotion']==1):
            y.append(data.loc[i,'emotion'])
            X.append(data[' pixels'][i].split(' '))
    X = np.array(X)
    X = X.astype('float64')
    X = np.divide(X,255.0)
    y = np.array(y)

    if(balance_ones):
        X0, y0 = X[y==0], y[y==0]
        X1 = X[y==1]
        X1 = np.repeat(X1,len(X0)/len(X1),axis=0)
        X = np.vstack([X0,X1])
        y = np.array([0]*len(X0)+[1]*len(X1))
    return X,y

#TESTING
def main():
   X,y= getBinaryData()
   print(X.shape)
   print(y.shape)



if __name__ == '__main__':
   main()
