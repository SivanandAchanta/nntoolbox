clear all; close all; clc;


% split train test val
text_fulldir = '../../../feats/lmag/';
audio_fulldir = '../../../feats/irm/';

files = dir(strcat(audio_fulldir,'*.irm'));

text_traindir = '../../../data/train/lmag/';
audio_traindir = '../../../data/train/irm/';

text_valdir = '../../../data/val/lmag/';
audio_valdir = '../../../data/val/irm/';

text_testdir = '../../../data/test/lmag/';
audio_testdir = '../../../data/test/irm/';

mkdir(audio_traindir);
mkdir(audio_valdir);
mkdir(audio_testdir);

mkdir(text_traindir);
mkdir(text_valdir);
mkdir(text_testdir);

aext = '.irm';
text = '.lmag';

% ivecv = [11:20:length(files)];
% ivect = [11:40:length(files)];
% 
% fid = fopen('../../../feats/filter_out.txt','r');
% M = textscan(fid,'%s');
% M = M{1};
% fclose(fid);
% 
% splittraintest


% make val data
matpath = '../../../matfiles_16KHz/';
mkdir(matpath);


files = dir(strcat(text_valdir,'*',text));
data = [];
targets = [];
clv = [];

makevaldata

% make test data

files = dir(text_testdir);
data = [];
targets = [];
clv = [];

maketestdata


% make train data

numfiles_batch = 800;
files = dir(text_traindir);
num_bats = ceil((length(files)-2)/numfiles_batch);
fprintf('total number of training batches %d \n',num_bats);

maketraindata
