% Step 0: Set model name
model_name = 'hl' % name of the model
model_dir = strcat('../basic/',model_name,'/');

% Step 1: Add paths
addpath(genpath('../../../../general_neuralnet_modules/'));
addpath(genpath(model_dir));

% Step 2: NN params settings
numepochs = 10 % number of epochs
gpu_flag = 0 % set the flag to 0 to run on CPU
sgd_type = 'sgdcm' % (adam/adadelta/sgdcm)
arch_name1 = '257N257N' % architecture
ol_type = 'L' % ouput layer type ( Usually 'L' (Linear) for Regression Problems and 'M' (Softmax) for Classification Problems)
cfn = 'ls' % cost function 'nll' for calssification and 'ls' for regression
wtinit_meth = 'gi' % gi/ui - Gaussian init/ Uniform init
check_valfreq = 10 % check validtion error for every "check_valfreq" minibats
gradCheckFlag = 0 % check back-prop with numerical gradients
train_test_ratio = 10 % the ratio of train set to be tested during model validation 
gcth_vec = [1] % gradient clipping hyperparameters
l1_vec = [0] % l1 norm regularization 
l2_vec = [0] % l2 norm regularization 
dp = [0.5] % dropout vec

% Step 3: Weight initialization hyper parameters (This is model specific, please see the model dir)
wtinit_hyperparam_init

% Step 4: Sgd hyper param init (This is model independent)
sgd_hyperparam_init

% Step 5: Set input/output dimensions along with dimensions across which normalization has to take place (This is experiment dependent)
set_io_dimensions

% Step 6: Input/Output normalization settings
in_nml_meth = 'mvni' % input nml meth ('mvni'/'mmni'/'')
out_nml_meth = 'mvno'  % output nml meth ('mvni'/'mmni'/'')
set_io_nml

% Step 7: Set the data directory (directory where the matfiles are present)
datadir = '../../../matfiles_16KHz/'

% Step 8: Make directories to write parameter files , error per epoch and average lengths of gradients
wtdir = '../../../wt/';
errdir = '../../../err/';

mkdir(wtdir);
mkdir(errdir);

% Step 9: Set variables for gradient checking
if gradCheckFlag    
    gc_flag = 0;
    sl = 10;
else
    gc_flag = 0;
end

% Step 10: Do nothing !!!
files = dir(strcat(datadir,'train*.mat'));
nb = length(files);
clear files

files = dir(strcat(wtdir,'W_*'));
nwt = length(files);
clear files
