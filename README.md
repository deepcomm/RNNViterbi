# Decoding Convolutional Codes

This is an example code for [Lecture 1](https://deepcomm.github.io/jekyll/pixyll/2020/02/01/learning-viterbi/) in our blog [deepcomm.github.io](https://deepcomm.github.io).  
Reference paper: [Communication Algorithms via Deep Learning, ICLR 2018](https://openreview.net/pdf?id=ryazCMbR-)

### Installation of tensorflow and Keras (assuming you have python3): 
``
pip3 install tensorflow; pip3 install keras
``

### To execute testing with a pre-trained model we provide (TrainedModel.h5),  
``
python3 main.py 
``

### To execute testing with your own trained model, 
``
python3 main.py --path_trained_model "MyTrainedModel.h5"
``

### To execute the training and testing, run 
``
python3 main.py -if_training True
``

### Tested on the following environment 
Python version 3.7.3
Tensorflow version 1.14.0
Keras version 2.3.1
