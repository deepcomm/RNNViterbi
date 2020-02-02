# RNNViterbi

Reference: Communication Algorithms via Deep Learning, ICLR 2018

### To execute testing with a pre-trained model, run 

python3 main.py --path_trained_model "MyTrainedModel.h5"

python3 main.py (without a specified path) will load the default model we provide (TrainedModel.h5)

### To execute the training and testing, run 

python3 main.py -if_training True

### Tested on the following environment 
Python version 3.7.3 
Tensorflow version 1.14.0
Keras version 2.3.1
