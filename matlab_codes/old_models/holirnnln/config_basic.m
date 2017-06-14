addpath(genpath('../general_neuralnet_modules/'));

% NN params settings
numepochs = 10
gpu_flag = 0 % Set the flag to 0 to run on CPU
sgd_type = 'adadelta' % (adam/adadelta/sgdcm)
arch_name1 = '6R' % Architecture
ol_type = 'L' % Ouput Layer Type ( Usually 'L' (Linear) for Regression Problems and 'M' (Softmax) for Classification Problems)
cfn = 'ls' % cost function 'nll' for calssification and 'ls' for regression
wtinit_meth = 'di'
check_valfreq = 10 % check validtion error for every "check_valfreq" minibats
model_name = 'holirnnln'
gradCheckFlag = 1;

l1_vec = [0];
l2_vec = [0];

switch sgd_type
    case 'sgdcm'
        lr_vec = [1e-1 1e-2];
        mf_vec = [0.0];
    case 'adadelta'
        rho_vec = [0.98];
        eps_vec = [1e-8 1e-9];
        mf_vec = [0];
    case 'adam'
        alpha_vec = [1e-3 1e-4];
        beta1_vec = [0.9];
        beta2_vec = [0.999];
        eps_hp = 1e-6;
        lam = 1 - eps_hp;
end

% make directories to write parameter files , error per epoch and average lengths of gradients
datadir = '../matfiles_16KHz/';
feat_name = 'cmp'; 
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
    invec = [1:5];
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

% Initialization hyper parameters
si = 0.1;
ri = 0.1;
so = 0.1;

% gradient clipping hyperparameters
gcth_vec = [1];
if gradCheckFlag
    gc_flag = 0;
else
    gc_flag = 1;
end

% set params of nonlinearity
a_tanh = 1.7159;
b_tanh = 2/3;
bby2a = (b_tanh/(2*a_tanh));
