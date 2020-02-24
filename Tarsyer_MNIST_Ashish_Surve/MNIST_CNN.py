"""
This program classfies the digit in MNIST dataset using a CNN.
Some noise is added to digits using gaussian distribution.

https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
class CNN:
    """
    x_train : 60000 data
    y_train :  label of the digit
    x_test : for testing
    y_test : for testing labels
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # @staticmethod
    # def normaliseAndReshape():
    #     CNN.x_train = CNN.x_train.reshape(CNN.x_train.shape[0], 28, 28, 1)
    #     CNN.x_test = CNN.x_test.reshape(CNN.x_test.shape[0], 28, 28, 1)
    #     input_shape = (28, 28, 1)
    #     # Making sure that the values are float so that we can get decimal points after division
    #     CNN.x_train = CNN.x_train.astype('float32')
    #     CNN.x_test = CNN.x_test.astype('float32')
    #     # Normalizing the RGB codes by dividing it to the max RGB value.
    #     CNN.x_train /= 255
    #     CNN.x_test /= 255
    #     print('x_train shape:', CNN.x_train.shape)
    @staticmethod
    def TrainGaussCNN(Xx_train,Xx_test):
        # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=Xx_train.shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dropout(0.1))
        model.add(Dense(28*28, activation=tf.nn.softmax))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x=Xx_train, y=CNN.x_train, epochs=5)

        print(model.evaluate(Xx_test, CNN.x_test))
        mage_index = 4444
        plt.imshow(Xx_test[image_index].reshape(28, 28), cmap='Greys')
        plt.show()

    @staticmethod
    def addGaussianNoise(mean=0,var=0.1):
        """
        :param mean: mean for gaussian
        :param var: variance for gaussian
        :return: a gaussed dataset
        """
        # for training data
        no_of_images, row, col = CNN.x_train.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (no_of_images, row, col))
        gauss = gauss.reshape(no_of_images, row, col)
        #plt.imshow(CNN.x_train[image_index], cmap='Greys')
        Xx_train = CNN.x_train + gauss

        # for testing data
        no_of_images, row, col = CNN.x_test.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (no_of_images, row, col))
        gauss = gauss.reshape(no_of_images, row, col)
        # plt.imshow(CNN.x_train[image_index], cmap='Greys')
        Xx_test = CNN.x_test + gauss
        return Xx_train, Xx_test



if __name__=="__main__":
    print("No of Samples: ",CNN.x_train.shape[0])

    image_index=1313
    print("X_train : ", CNN.x_train.shape)
    print("X_test : ", CNN.x_test.shape)
    #print(type(CNN.y_train))      # <class 'numpy.ndarray'>
    #plt.imshow(CNN.x_train[image_index],cmap='Greys')
    #plt.show()
    Xx_train, Xx_test = CNN.addGaussianNoise(0.2, 0.4)
    CNN.TrainGaussCNN(Xx_train, Xx_test)




