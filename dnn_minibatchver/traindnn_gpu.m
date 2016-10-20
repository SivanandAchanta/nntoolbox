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
best_val_loss = inf;
best_iter = 0;
num_up = 0;

trainerr = 0;
valerr = 0;
testerr = 0;

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
        [X,Y,bs] = get_XY(train_batchdata, train_batchtargets, rp, i, gpu_flag);
        
        % fp
        [otl] = get_otl(bs,nl,nlv);
        [ol] = fpav_gpu(X,GW,Gb,nl,fl,nh,wtl,btl,a_tanh,b_tanh,bs);
        
        % bp
        [gW,gb] = bpav_gpu(X,Y,ol,GW,otl,btl,wtl,a_tanh,b_tanh,bby2a,fl,bs,nl,nh,cfn,l1,l2);

        % *** gradCheck ***
        % gradCheck
        
        % Update Params using Appropriate SGD Method        
        update_params
        
        if mod(num_up,check_valfreq) == 0
            
            tic
            [trainerr] = compute_error(train_batchdata,train_batchtargets,train_clv,train_test_numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh);
            toc
            
            % Validation data error computation
            tvde = tic;
            [valerr] = compute_error(val_batchdata,val_batchtargets,val_clv,val_numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh);
            toc(tvde)
            
            % Print error (validation) per epoc
            fprintf('Epoch : %d  Update : %d Train Loss : %f Val Loss : %f \n',NE,num_up,trainerr,valerr);
            
            if valerr < best_val_loss
                if valerr < (best_val_loss*imp_thresh)
                    patience = max(patience,iter*patience_inc);
                end
                best_val_loss = valerr;
                best_iter = iter;
                
                [testerr] = compute_error(test_batchdata,test_batchtargets,test_clv,test_numbats,gpu_flag,GW,Gb,nl,nlv,fl,nh,wtl,btl,cfn,a_tanh,b_tanh);
                
                % Print error (testing) per epoc
                fprintf('\t Epoch : %d  Update: %d Test Loss : %f \n',NE,num_up,testerr);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%% save weight file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % save parameters every epoch
                W = gather(GW); b = gather(Gb);
                save(strcat(wtdir,'W_',arch_name,'.mat'),'W','b');
            end
            
            % Print error (validation and testing) per epoc
            fprintf(fid,'%d %d %f %f %f \n',NE,num_up,trainerr,valerr,testerr);
            
        end
        
        if isnan(valerr) || isnan(testerr)
            break;
        end
        
    end
    toc(twu)
    
    if isnan(valerr) || isnan(testerr)
        break;
    end
    
    
end

fclose(fid);

fprintf('Training done !!!\n')
fprintf('Best val error : %f ; at epoch : %d ; with test error : %f \n', best_val_loss,floor(best_iter/train_numbats),testerr)

