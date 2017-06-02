% NN params settings
numepochs = 10;
gpu_flag = 0 % Set the flag to 0 to run on CPU
sgd_type = 'sgdcm' % (adam/adadelta/sgdcm)
arch_name1 = '6R8R' % Architecture
ol_type = 'M' % Ouput Layer Type ( Usually 'L' (Linear) for Regression Problems and 'M' (Softmax) for Classification Problems)
cfn = 'nll' % cost function 'nll' for calssification and 'ls' for regression
wtinit_meth = 'rg'
check_valfreq = 10 % check validtion error for every "check_valfreq" minibats
model_name = 'dnnwa'
gradCheckFlag = 1;
l1 = 0;
l2_vec = [0];
train_test_ratio = 10;
