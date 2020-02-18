
from matplotlib.colors import ListedColormap

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


if __name__=="__main__":
    import pandas as pd

    # get dataset from site

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/iris/iris.data',
                     header=None)

    import matplotlib.pyplot as plt
    import numpy as np

    # select setosa and vericolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
                                                                # -1=Vericose
                                                                # 1= Iris-setosa
    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values
    '''
    # plot data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='vericolor')
    # plt.scatter([2.5,5.6],color='green',marker='v',label='predict')

    plt.xlabel('sepal length(cm)')
    plt.ylabel('petal length(cm)')
    plt.legend(loc='upper left')
    #plt.show()                                                 **dataser show
    '''
    # =====================================================
    # Train Adaline 1
    # =====================================================
    # Starting with Adaline
    from AdalineGD import AdalineGD

    agd = AdalineGD(eta=0.1, n_iter=10)
    agd.fit(X, y)
    # =====================================================
    # show lepoch vs iters
    '''
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ax[0].plot(range(1,len(agd.cost_)+1),np.log10(agd.cost_),marker='o')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')


    # =====================================================
    # Train Adaline 2
    # =====================================================
    # Starting with Adaline
    agd2 = AdalineGD(eta=0.0001, n_iter=10)
    agd2.fit(X, y)
    # =====================================================
    ax[1].plot(range(1, len(agd2.cost_) + 1), agd2.cost_, marker='o')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    #plt.show()
    '''
    # =====================================================
    # Train Adaline 2
    # =====================================================
    # Starting with Adaline
    agd3 = AdalineGD(eta=0.01, n_iter=100)

    X_std=np.copy(X)                                                # **Standarization using
    X_std[:,0]=(X[:,0] - X[:,0].mean())/ X[:,0].std()               #  means and std deviation
    X_std[:,1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    agd3.fit(X_std, y)
    # =====================================================

    plot_decision_regions(X_std,y,classifier=agd3)                  # ** plot std data
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(agd3.cost_) + 1), agd3.cost_, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel('Sum-squared-error')
    plt.show()
