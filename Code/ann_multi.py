import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getData,softmax,cost2,y2indicator,error_rate,init_weight_and_bias,relu,tanh

class ANN:
    def __init__(self,M):
        self.M = M

    def fit(self,X,Y,learning_rate=10e-6,regularisation=10e-1,epochs=10000,show_fig=False):
        X,Y = shuffle(X,Y)

        # print("X.shape"+str(X.shape))
        # print("Y.shape"+str(Y.shape))
        Xvalid, Yvalid = X[-1000:],Y[-1000:]
        # Tvalid = y2indicator(Yvalid) # WE DONT NEED TVALID CAUSE WE ARE USING COST2
        X,Y = X[:-1000],Y[:-1000]
        # print("X.shape"+str(X.shape))
        # print("Y.shape"+str(Y.shape))
        N,D = X.shape
        K = len(set(Y))
        T = y2indicator(Y) #Need this for gradient descent


        self.W1,self.b1 = init_weight_and_bias(D,self.M)
        self.W2,self.b2 = init_weight_and_bias(self.M,K)


        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation
            pY,Z = self.forward(X)

            # gradient descent
            pY_T = pY - T
            self.W2 -= learning_rate*(Z.T.dot(pY_T) + regularisation*self.W2)
            self.b2 -= learning_rate*((pY_T).sum(axis=0) + regularisation*self.b2)

            # dZ = pY_T.dot(self.W2.T) * (Z>0) #Relu
            dZ = pY_T.dot(self.W2.T) * (1- Z*Z) # Tanh
            self.W1 -= learning_rate*(X.T.dot(dZ) + regularisation*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + regularisation*self.b1)

            if i%10 ==0 :
                pYvalid,_ = self.forward(Xvalid)
                c = cost2(Yvalid,pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid,axis=1))
                print("i : "+str(i)+"; Cost : "+str(c)+"; Error : "+str(e))
                if e < best_validation_error:
                    best_validation_error = e

        print("Best Validation error : "+str(best_validation_error))

        if(show_fig):
            plt.plot(costs)
            plt.show()


    def forward(self,X):
        #Z = relu(X.dot(self.W1)+self.b1)
        Z = tanh(X.dot(self.W1)+self.b1)
        return softmax(Z.dot(self.W2)+self.b2), Z

    def predict(self,X):
        pY,_ = self.forward(X)
        return np.argmax(pY,axis=1)

    def score(self,X,Y):
        prediction = self.forward(X)
        return (1 - error_rate(Y,prediction))






def main():
    X, Y = getData()
    model = ANN(100)
    model.fit(X,Y,show_fig = True)
    print(model.score(X,Y))


if __name__ == '__main__':
    main()
