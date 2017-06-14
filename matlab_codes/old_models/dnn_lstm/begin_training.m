
% weight initialization
if dnn_flag
    wt_init_dnn
end
wt_init

% get full arch_name
get_fullarchname

% train the model
train_dnn_lstm