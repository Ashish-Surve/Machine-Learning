
#adaline using batch gradient descent
import numpy as np

class LogisticRegressionGD(object):
    """ADAptive LInear NEuron classfier
    Parameters
    ------------
    eta : float
        learning rate
    iter : int
        Passes over the training dataset
    random_state : int
        Random number generator seed

     Attributes
     -------------
    w_ : 1d array
        weights after fitting
    error_ : list
        Number of misclassfication(updates) in each epoch.
    """
    def __init__(self,eta=0.05,n_iter=100,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,X,y):
        """
        Parameters
        ------------
        :param X: {array-like},shape =[n_smaples,n_features]
        :param y: array-like,shape=[n_shape]

        Returns
        --------------
        self : object
        """

        rgen=np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_ = []
        # adaptive learning rate
        alr=self.eta             # for normal replace with eta fixed learning rate
        for _ in range(self.n_iter):
            net_input=self.net_input(X)
            output = self.activation(net_input)
            errors=(y-output)
            #self.w_[1:]+=alr*X.T.dot(errors)               Adaptive Learning rule
            self.w_[1:] += self.eta * X.T.dot(errors)
            #self.w_[0]+=alr*errors.sum()                   Adaptive Learning rule
            self.w_[0] += self.eta * errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
            #alr=1/(self.n_iter+0.5)                        Adaptive Learning rule
        return self

    def net_input(self,X):
        """Calculate net input
        Theta=wTx
        ** the bias is added at end here.
        """
        return np.dot(X,self.w_[1:])+self.w_[0]

    def activation(self,z):
        """Linear Activation"""
        return 1./(1. +np.exp(-np.clip(z,-250,250)))

    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.net_input(X)>=0.0,1,0)

