addpath(genpath('../../general_neuralnet_modules/'));
addpath('fp_bp_fns/');
addpath('gradcheck_fns/');

% NN params settings
numepochs = 5
gpu_flag = 0 % Set the flag to 0 to run on CPU
sgd_type = 'sgdcm' % (adam/adadelta/sgdcm)
arch_name1 = '5R' % Architecture
ol_type = 'L' % Ouput Layer Type ( Usually 'L' (Linear) for Regression Problems and 'M' (Softmax) for Classification Problems)
cfn = 'ls' % cost function 'nll' for calssification and 'ls' for regression
wtinit_meth = 'gi'
check_valfreq = 10 % check validtion error for every "check_valfreq" minibats
model_name = 'll'
gradCheckFlag = 1;
dp = [0.5];

l1_vec = [0];
l2_vec = [0];

sgd_hyperparam_init
wtinit_hyperparam_init

% make directories to write parameter files , error per epoch and average lengths of gradients
datadir = '../matfiles_16KHz/';
feat_name = 'cmp'; 
wtdir = '../../wt/';
errdir = '../../err/';


files = dir(strcat(datadir,'train*.mat'));
nb = length(files);
clear files

files = dir(strcat(wtdir,'W_*'));
nwt = length(files);
clear files

% Set the input output dimensions and the normalization flags
if strcmp(feat_name,'mgc')
   outvec = [1:150];
elseif strcmp(feat_name,'f0')
   outvec = [232:235];
elseif strcmp(feat_name,'bap')
   outvec = [151:228];
elseif strcmp(feat_name,'cmp')
   outvec = [1:235];
end

invec = [1:347];
mvnivec = [303:339 343:347];

in_nml_meth = 'mvni'
out_nml_meth = 'mvno'

if gradCheckFlag
    invec = [1:3];
    outvec = [1:3];
    
    din = length(invec);
    dout = length(outvec);
    
end

if strcmp(in_nml_meth,'mvni')
   intmvnf = 1;
   intmmnf = 0;
elseif strcmp(in_nml_meth,'mmni')
   intmvnf = 0;
   intmmnf = 1;
else
   intmvnf = 0;
   intmmnf = 0;
end

if strcmp(out_nml_meth,'mvno')
   outtmvnf = 1;
   outtmmnf = 0;
elseif strcmp(out_nml_meth,'mmno')
   outtmvnf = 0;
   outtmmnf = 1;
else
   outtmvnf = 0;
   outtmmnf = 0;
end

lr = 0;
mf = 0;
rho_hp = 0;
eps_hp = 0;
alpha = 0;
beta1 = 0;
beta2 = 0;
