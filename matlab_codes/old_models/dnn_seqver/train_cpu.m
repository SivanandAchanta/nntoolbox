%{
###########################################################################
##                                                                       ##
##                                                                       ##
##                       IIIT Hyderabad, India                           ##
##                      Copyright (c) 2015                               ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  IIIT HYDERABAD AND THE CONTRIBUTORS TO THIS WORK                     ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL IIIT HYDERABAD NOR THE CONTRIBUTORS BE LIABLE                  ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
##          Author :  Sivanand Achanta (sivanand.a@research.iiit.ac.in)  ##
##          Date   :  Jul. 2015                                          ##
##                                                                       ##
###########################################################################
%}

% open error text file
fid = fopen(strcat(errdir,'/err_',arch_name,'.err'),'w');

% early stopping params (Theano DeepLearningTutorials)
patience = 10000;
patience_inc = 2;
imp_thresh = 0.995;
best_val_err = inf;
best_epoch = 0;
num_up = 0;
train_test_numbats = train_numbats/train_test_ratio;

train_err = 0;
val_err = 0;
test_err = 0;

for NE = 1:numepochs
    
    % for each epoch
    iter = (NE-1)*train_numbats;
    
    % Weight update step
    twu = tic;
    
    rp = randperm(train_numbats);
    
    % fp and bp for each batch
    for i = 1:train_numbats
        
        num_up = num_up + 1;
        
        % get data
        [X,Y,bs] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
        
        % fp
        [otl] = get_otl(bs,nl,nlv);
        [ol] = fpav_cpu(X,GW,Gb,nl,fl,nh,wtl,btl,bs);
        
        % bp
        [gW,gb] = bpav_cpu(X,Y,ol,GW,otl,btl,wtl,fl,bs,nl,nh,cfn,l1,l2);
        
        % *** gradCheck ***
        gradCheck
        
        % Update Params using Appropriate SGD Method
        update_params
        
        if mod(num_up,check_valfreq) == 0
            
            tic
            [train_err] = compute_error(train_batchdata,train_batchtargets,train_clv,train_test_numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh);
            toc
            
            % Validation data error computation
            tvde = tic;
            [val_err] = compute_error(val_batchdata,val_batchtargets,val_clv,val_numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh);
            toc(tvde)
            
            % Print error (validation) per epoc
            fprintf('Epoch : %d  Update : %d Train Loss : %f Val Loss : %f \n',NE,num_up,train_err,val_err);
            
            if val_err < best_val_err
                if val_err < (best_val_err*imp_thresh)
                    patience = max(patience,iter*patience_inc);
                end
                best_val_err = val_err;
                best_epoch = NE;
                
                [test_err] = compute_error(test_batchdata,test_batchtargets,test_clv,test_numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh);
                
                % Print error (testing) per epoc
                fprintf('\t Epoch : %d  Update: %d Test Loss : %f \n',NE,num_up,test_err);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%% save weight file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % save parameters every epoch
                W = (GW); b = (Gb);
                save_params
            end
            
            % Print error (validation and testing) per epoc
            fprintf(fid,'%d %d %f %f %f \n',NE,num_up,train_err,val_err,test_err);
            
        end
        
        if isnan(val_err) || isnan(test_err)
            break;
        end
        
    end
    
    fprintf('Time taken for Epoch : %d is %f sec \n',NE,toc(twu));
end

fclose(fid);

% print the best val and test error as well as the epoch at which it is obtained
fprintf('Best loss obtained at Epoch : %d; Val Loss: %f   Test Loss: %f \n',best_epoch,best_val_err,test_err);


