
% Add paths
addpath(genpath('../general_neuralnet_modules/'));
addpath(genpath('.'));

% Make directories to write parameter files , error per epoch
datadir = '../matfiles_12_39D/';

set_nnparams
set_io_nml
sgd_hyperparam_init
wtinit_hyperparam_init

wtdir = ('../wt/');
errdir =  ('../err/');
mkdir(wtdir);
mkdir(errdir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do not change the below parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir(strcat(datadir,'train*.mat'));
nb = length(files);
clear files

files = dir(strcat(wtdir,'W*.mat'));
nwt = length(files) + 1;
clear files
