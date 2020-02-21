"""
This code executes OvA LRGD on Iris Dataset
"""
from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#understand this and use it frequently
#helps to plot the decision bindary
def plot_decision_regions(X,y,classifier,resolution=0.02):
    #setup marker generator and color map
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min,x1_max=X[:,0].min() -1,X[:,0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=cl,edgecolor='black')


#splittin of test and train
"""
train_test_split
random_state: gives us control over seed so that we can reproduce results
stratify : puts same proportions of labels in test and train
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

X_train_01_subset=X_train[(y_train==0) | (y_train==1)]
y_train_01_subset=y_train[(y_train==0) | (y_train==1)]

from LogisticRegression.LogisticRegressionGD import LogisticRegressionGD

lrgd = LogisticRegressionGD(eta=0.05,
                            n_iter=1000,
                            random_state=1)

lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()







