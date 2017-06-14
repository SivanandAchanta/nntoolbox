% Purpose : Training module for DNN-LSTM

% open error file
fid = fopen(strcat(errdir,'err_',arch_name,'.err'),'w');

% early stopping params (Theano DeepLearningTutorials)
patience = 10000;
patience_inc = 2;
imp_thresh = 0.995;
val_freq = min(train_numbats,patience/2);
best_val_err = inf;
best_iter = 0;
num_up = 0;
train_test_numbats = round(train_numbats/train_test_ratio);

val_err = 0;
test_err = 0;
train_err = 0;


for NE = 1:numepochs
    
    % for each epoch
    iter = (NE-1)*train_numbats;
    
    % Weight update step
    twu = tic;
    
    rp = randperm(train_numbats);
    
    train
    
    if isnan(val_err) || isnan(test_err)
        break;
    end
    
    fprintf('Time taken for Epoch : %d is %f sec \n',NE,toc(twu));
end

fclose(fid);

% print the best val and test error as well as the epoch at which it is obtained
fprintf('Best loss obtained at Epoch : %d; Val Loss: %f   Test Loss: %f \n',best_epoch,best_val_err,test_err);
