"""
This program classfies the digit in MNIST dataset using a CNN.
Some noise is added to digits using gaussian distribution.

Ran this code on Google Collab to obtain the model file in .h5 format

https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
"""
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    ####

    # USAGE
    # python train_conv_autoencoder.py

    # set the matplotlib backend so figures can be saved in the background
    import matplotlib

    matplotlib.use("Agg")

    # import the necessary packages
    from convautoencoder import ConvAutoencoder
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.datasets import mnist
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse
    import cv2

    args = {}
    args["samples"] = 8
    args["output"] = "op.png"
    args["plot"] = "plot.png"

    # initialize the number of epochs to train for and batch size
    EPOCHS = 10
    BS = 32

    # load the MNIST dataset
    print("[INFO] loading MNIST dataset...")
    # ((trainX, _), (testX, _)) = mnist.load_data()
    trainX, testX = x_train, x_test

    # gauss nosie from keras documentation
    noise_factor = 0.5
    trainNoise = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    testNoise = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    trainXNoisy = np.clip(x_train_noisy, 0., 1.)
    testXNoisy = np.clip(x_test_noisy, 0., 1.)

    print(trainX.shape)

    #################################################################
    ## uncomment below line if we need to make the model file again
    ################################################################

    # print("[INFO] building autoencoder...")
    # (encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1)
    # opt = Adam(lr=1e-3)
    # autoencoder.compile(loss="mse", optimizer=opt)
    # # train the convolutional autoencoder
    # H = autoencoder.fit(
    #     trainXNoisy, trainX,
    #     validation_data=(testXNoisy, testX),
    #     epochs=EPOCHS,
    #     batch_size=BS)
    # # construct a plot that plots and saves the training history
    # N = np.arange(0, EPOCHS)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(N, H.history["loss"], label="train_loss")
    # plt.plot(N, H.history["val_loss"], label="val_loss")
    # plt.title("Training Loss and Accuracy")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig(args["plot"])
    #
    # print("[INFO] making predictions...")
    # decoded = autoencoder.predict(testXNoisy)
    # outputs = None
    # # loop over our number of output samples
    # for i in range(0, args["samples"]):
    #     # grab the original image and reconstructed image
    #     original = (testXNoisy[i] * 255).astype("uint8")
    #     recon = (decoded[i] * 255).astype("uint8")
    #     # stack the original and reconstructed image side-by-side
    #     output = np.hstack([original, recon])
    #     # if the outputs array is empty, initialize it as the current
    #     # side-by-side image display
    #     if outputs is None:
    #         outputs = output
    #     # otherwise, vertically stack the outputs
    #     else:
    #         outputs = np.vstack([outputs, output])
    # # save the outputs image to disk
    # cv2.imwrite(args["output"], outputs)
    #
    # autoencoder.save('CNN_AutoencoderV3_for_denoise.h5')

    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    adam = Adam(lr=0.01)

    # Load Noise Model
    from tensorflow.keras.models import load_model

    autoencoder = load_model('CNN_AutoencoderV3_for_denoise.h5')
    autoencoder_op = autoencoder.predict(x_train_noisy)
    autoencoder_op.shape = (autoencoder_op.shape[0], 784)
    print(autoencoder_op.shape)
    # Load Dense Model
    import Tarsyer_MNIST_Ashish_Surve.Train_Digits_MNIST as TDM

    dense_model = TDM.ConvMNIST.build(784)

    # Train Dense
    dense_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    x_test_a = autoencoder.predict(x_test)
    x_test_a.shape = (10000, 784)
    num_digits = 10
    from tensorflow.keras.utils import to_categorical

    y_binary = to_categorical(y_train, num_digits)
    y_binary_2 = to_categorical(y_test, num_digits)
    history1 = dense_model.fit(autoencoder_op, y_binary,
                               batch_size=batch_size,
                               epochs=training_epochs,
                               verbose=2,
                               validation_data=(x_test_a, y_binary_2))

    # # construct a plot that plots and saves the training history
    N = np.arange(0, training_epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history1.history["loss"], label="train_loss_dense")
    plt.plot(N, history1.history["val_loss"], label="val_loss_dense")
    plt.title("Training Loss and Accuracy on Dense")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("Dense_plot.png")

    dense_model.save('DenseModelFINAL.h5')
