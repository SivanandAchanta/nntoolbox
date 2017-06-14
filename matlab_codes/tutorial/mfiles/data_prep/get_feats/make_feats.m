clear all; close all; clc;

in_dir = '../../feats/lmag/';
out_dir = '../../feats/irm/';
mkdir(in_dir);
mkdir(out_dir);

N = 100;
din = 257;
dout = 257;


for i = 1:10
    
    X = randn(N,din);
    Y = randn(N,dout);
    
    dlmwrite(strcat(in_dir,num2str(i),'.lmag'),sinlge(X),'delimiter',' ');
    dlmwrite(strcat(out_dir,num2str(i),'.irm'),sinlge(Y),'delimiter',' ');
    
end




