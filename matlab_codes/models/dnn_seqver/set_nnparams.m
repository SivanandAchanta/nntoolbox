% NN params settings
numepochs = 20; % Number of Epochs
gpu_flag = 0; % Set the flag to 0 to run on CPU
sgd_type = 'adadelta'; % (adam/adadelta/sgdcm)
arch_name1 = '5R'; % Architecture
ol_type = 'L'; % Ouput Layer Type ( Usually 'L' (Linear) for Regression Problems and 'M' (Softmax) for Classification Problems)
cfn = 'ls';  % Cost-function 'nll' (negative log-likelihood) for calssification and 'ls' (least squares) for regression
wtinit_meth = 'yi'; % 'yi - yoshua init, rw - random walk init , si - sparse init'
l1 = 0; % l1 regularization penalty coefficent settings
l2_vec = [0]; % l2 regularization penalty coefficent settings
check_valfreq = 1000; % check validation loss for every "N" updates
gradCheckFlag = 1;
model_name = 'dnn';
train_test_ratio = 10;