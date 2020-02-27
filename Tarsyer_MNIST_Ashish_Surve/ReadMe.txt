Execute MNIST_CNN.py to see the output.

How is the program implemented.

1. created a model .h5 (CNN_AutoencoderV3_for_denoise.h5) for the autoencoder

This model was made on GoogleCollab for faster training(GPU runtime).

2. Created another Dense model and fed the output of the autoencoder to the dense model, as it is simpler and efficent than freezing the autoencoder and adding Dense layers.

3. The loss/ epoch graph is saved in Dense_plot.png,
The output result is stored in Result_Final.txt
