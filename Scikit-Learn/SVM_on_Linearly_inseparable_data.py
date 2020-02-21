"""
SVM for linearly inseparable data
Data is created by xor of numpy

Radial Basis Function (RBF) kernel used here
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()

from sklearn.svm import SVC
from HelperFunctions.Plotting import plot_decision_regions1

# using rbf kernel instead of linear
"""
gamma:
    can be understood as a cut-off
    parameter for the Gaussian sphere. If we increase the value for , we increase the
    influence or reach of the training samples, which leads to a tighter and bumpier
    decision boundary.
C:
    Inverse Regularisation parameter    
"""
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions1(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# =================================================================================
# using the RBF kernel on Iris dataset with small gamma

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# splitting of test and train
"""
train_test_split
random_state: gives us control over seed so that we can reproduce results
stratify : puts same proportions of labels in test and train
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
"""
Standard Scaler
fit : estimates mean and Std Deviation
transform : standardise the data-set
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

"""
Try gamma as 100 then 5,
Inference is that higher gamma tighten the boundary
which is not good.
"""
svm = SVC(kernel='rbf', random_state=1, gamma=3, C=1.0)
svm.fit(X_train_std, y_train)
from HelperFunctions.Plotting import plot_decision_regions2

plot_decision_regions2(X_combined_std,
                       y_combined, classifier=svm,
                       test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

y_pred = svm.predict(X_test_std)
print("Misclassified samples: %d " % (y_test != y_pred).sum())

# use metrics module a lot.
from sklearn.metrics import accuracy_score
print("Accuracy: %.2f " % accuracy_score(y_test, y_pred))
