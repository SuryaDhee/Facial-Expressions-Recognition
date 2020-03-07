import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData,sigmoid,sigmoid_cost,relu,error_rate,init_weight_and_bias,tanh

class ANN:
    def __init__(self,M):
        self.M = M

    def fit(self,X,Y,learning_rate=5e-7,regularisation=1.0,epochs=10000,show_fig=False):
        X,Y = shuffle(X,Y)
        Y = np.reshape(Y,(len(Y),1)) #s
        # print("X.shape"+str(X.shape))
        # print("Y.shape"+str(Y.shape))
        Xvalid, Yvalid = X[-1000:],Y[-1000:]
        X,Y = X[:-1000],Y[:-1000]
        # print("X.shape"+str(X.shape))
        # print("Y.shape"+str(Y.shape))
        N,D = X.shape
        self.W1,self.b1 = init_weight_and_bias(D,self.M) #s
        self.W2,self.b2 = init_weight_and_bias(self.M,1) #s
        # self.W1 = np.random.randn(D, self.M) / np.sqrt(D) #lp
        # self.b1 = np.zeros(self.M) #lp
        # self.W2 = np.random.randn(self.M) / np.sqrt(self.M) #lp
        # self.b2 = 0 #lp

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation
            pY, Z = self.forward(X)

            # gradient descent
            pY_Y = pY - Y
            # print("X.shape"+str(X.shape))
            # print("pY.shape"+str(pY.shape))
            # print("Y.shape"+str(Y.shape))
            # print("Z.shape"+str(Z.shape))
            # print("W2.shape"+str(self.W2.shape))
            # print("pY_Y.shape"+str(pY_Y.shape))
            self.W2 -= learning_rate*(Z.T.dot(pY_Y) + regularisation*self.W2)
            self.b2 -= learning_rate*(pY_Y.sum() + regularisation*self.b2)
            dZ = pY_Y.dot(self.W2.T) * (Z>0) #s
            # dZ = np.outer(pY_Y, self.W2) * (Z > 0) #lp

            self.W1 -= learning_rate*(X.T.dot(dZ) + regularisation*self.W1)
            self.b1 -= learning_rate*(np.sum(dZ,axis=0) + regularisation*self.b1)

            if i%20 ==0 :
                pYvalid ,_ = self.forward(Xvalid)
                # print("Yvalid.shape"+str(Yvalid.shape))
                # print("pYvalid.shape"+str(pYvalid.shape))
                c = sigmoid_cost(Yvalid,pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
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
        # print("Z.shape"+str(Z.shape))
        # print("self.W2.shape"+str(self.W2.shape))
        # print("self.b2.shape"+str(self.b2.shape))
        ret =  sigmoid(Z.dot(self.W2)+self.b2)
        # print("ret.shape"+str(np.array(ret).shape))
        return ret, Z

    def predict(self,X):
        pY,_ = self.forward(X)
        return np.round(pY)

    def score(self,X,Y):
        prediction = self.forward(X)
        return (1 - error_rate(Y,prediction))*100






def main():
    X, Y = getBinaryData()
    # print("X.shape"+str(X.shape))
    # print("Y.shape"+str(Y.shape))
    # X0 = X[Y==0]
    # X1 = X[Y==1]
    # print("X0.shape"+str(X0.shape))
    # print("X1.shape"+str(X1.shape))

    model = ANN(100)
    model.fit(X,Y,show_fig = True)


if __name__ == '__main__':
    main()
