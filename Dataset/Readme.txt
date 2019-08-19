1. load('data_train.mat) will load training data structures to workspace. Each cell is a time series of vibration data.
2. load('data_train_labels.mat) will load the labels of training data. 1 -normal; 2-roller; 3-inner; 4-outer; 5-inner+roller; 6-outer+inner; 7-outer+inner+roller; 8-outer+roller;
3. load('data_test.mat) will load testing data structures to workspace. Each cell is a time series of vibration data.

Task: 
1. train a SOM using data_train and data_train_lables, visualize in health map;
2. predict the labels of data_test.

Other info:

Fs = 50000;   % Sampling rate
fr = 800/60;  % Spindle's rotating frequency (in Hz)
BPFO = 7.14*fr;  % outer defect freq.(in Hz)
BPFI = 9.88*fr;  % inner
BSF = 5.824*fr;  % roller