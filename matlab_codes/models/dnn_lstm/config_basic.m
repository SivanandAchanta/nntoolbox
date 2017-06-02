addpath(genpath('../../general_neuralnet_modules/'));
addpath('configninit_fns/');
addpath('fp_bp_fns/');
addpath('gradcheck_fns/');

% NN params settings
numepochs = 10
gpu_flag = 0 % Set the flag to 0 to run on CPU
sgd_type = 'sgdcm' % (adam/adadelta/sgdcm)
arch_name_dnn = '10N10N' % Architecture
arch_name1 = '10N' % Architecture
ol_type = 'L' % Ouput Layer Type ( Usually 'L' (Linear) for Regression Problems and 'M' (Softmax) for Classification Problems)
cfn = 'ls' % cost function 'nll' for calssification and 'ls' for regression
wtinit_meth = 'di'
wtinit_meth_dnn = 'yi'
check_valfreq = 10 % check validtion error for every "check_valfreq" minibats
model_name = 'dnn_lstm'
gradCheckFlag = 1
train_test_ratio = 10
dnn_flag = 1
l1 = 0
l2 = 0

% sgd hyper param init
sgd_hyperparam_init

% input/output normalization settings
in_nml_meth = '' % input nml meth ('mvni'/'mmni'/'')
out_nml_meth = ''  % output nml meth ('mvni'/'mmni'/'')
set_io_nml

% weight initialization hyper parameters
wtinit_hyperparam_init

% set synthetic data dimensions for gradient checking
if gradCheckFlag    
    invec = [1:5];
    outvec = [1:3];    
    din = length(invec);
    dout = length(outvec);    
    gc_flag = 0;
else
    gc_flag = 1;
end

% make directories to write parameter files , error per epoch and average lengths of gradients
datadir = '../';
wtdir = '../../wt/';
errdir = '../../err/';

mkdir(wtdir);
mkdir(errdir);

files = dir(strcat(datadir,'train*.mat'));
nb = length(files);
clear files

files = dir(strcat(wtdir,'W_*'));
nwt = length(files);
clear files

