% Purpose : Training module for RNN sequence generator

% open error file
fid = fopen(strcat(errdir,'err_',arch_name,'.err'),'w');

% early stopping params (Theano DeepLearningTutorials)
patience = 10000;
patience_inc = 2;
imp_thresh = 0.995;
val_freq = min(train_numbats,patience/2);
best_val_loss = inf;
best_iter = 0;
num_up = 0;

valerr = 0;
testerr = 0;
trainerr = 0;


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
        [X,Y,sl] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
        
        % Forward Pass
        [zm,im,fm,cm,om,hcm,hm,ym] = fp_lstm(X,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu);
        
        % Error at Output Layer
        switch cfn
            case 'ls'
                E   = -(Y - ym)/sl;
            case  'nll'
                E   = -(Y - ym)/sl;
        end
        
        % Backprop
        gU = (E'*hm);
        gbu = sum(E)';
        Eb = U'*E';
        
        [dhm,dom,dcm,dfm,dim,dzm] = bp_lstm(Eb,Rz,Ri,pi,Rf,pf,Ro,po,zm,im,fm,cm,om,hcm,nl,sl);
        [gWz,gRz,gbz,gWi,gRi,gpi,gbi,gWf,gRf,gpf,gbf,gWo,gRo,gpo,gbo] = gradients_lstm(X,hm,cm,dom,dfm,dim,dzm,nl);
        
        
        if gradCheckFlag
            gradCheck
        end
        
        % Gradient Clipping
        if gc_flag
            [gWz]  = gc(gWz,gcth);
            [gRz]  = gc(gRz,gcth);
            [gWi]  = gc(gWi,gcth);
            [gRi]  = gc(gRi,gcth);
            [gWf]  = gc(gWf,gcth);
            [gRf]  = gc(gRf,gcth);
            [gWo]  = gc(gWo,gcth);
            [gRo]  = gc(gRo,gcth);
            [gU]   = gc(gU,gcth);
        end
        
        % Update Params
        update_params
        
        % Check Validation Loss
        if mod(num_up,check_valfreq) == 0
            
            tic
            [val_err] = compute_error(val_batchdata,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu,val_numbats,cfn);
            toc
            
            fprintf('Epoch: %d, Val Loss : %f',NE,val_err);
            
            if val_err < best_val_loss
                
                best_val_loss = val_err;
                
                tic
                [test_err] = compute_error(test_batchdata,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu,test_numbats,cfn);
                toc
                
                % Print error (testing) per epoc
                fprintf('\t Epoch : %d  Update: %d Test Loss : %f \n',NE,num_up,test_err);                
                
                % save weight file
                save(strcat(wtdir,'W_',arch_name,'.mat'),'Wi','Ri','bi','pi','Wf','Rf','bf','pf','Wo','Ro','bo','po','Wz','Rz','bz','U','bu');
            end
            
            % Print error (validation and testing) per epoc
            fprintf(fid,'%d %d %f %f %f \n',NE,num_up,val_err,test_err);
        end
        
    end
    
    fprintf('Time taken for Epoch : %d is %f sec ... \n',NE,toc(twu));
end


fclose(fid);