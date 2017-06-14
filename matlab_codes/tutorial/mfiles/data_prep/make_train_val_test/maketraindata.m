% Purpose : make train data

for nb = 1:num_bats
    
    data = [];
    targets = [];
    clv = [];
    cnt = 0;
    
    efno = (numfiles_batch*(nb)+2);
    if efno > length(files)
        efno = length(files);
    end
    
    for i = (numfiles_batch*(nb-1)+1+2):efno
        
        [str,tok] = strtok(files(i).name,'.');
        fprintf('Processing Batch: %d, batch utt no. %d, utt %s for traning data \n',nb,length(clv),str);
        
        T = dlmread(strcat(text_traindir,files(i).name));
        A = dlmread(strcat(audio_traindir,str,aext));
        
        [tN,td] = size(T);
        [aN1,ad1] = size(A);
        
        nframes = min([tN aN1]);
        
        T = T(1:nframes,:);
        A = A(1:nframes,:);
        
        T(isnan(T)) = 0;
        
        if sum(sum(isnan(T)))
            fprintf('File %s has NaN elements in Text Feats\n',str)
            break
        end
        
        if sum(sum(isinf(T)))
            fprintf('File %s has Inf elements in Text Feats\n',str)
            break
        end
        
        if sum(sum(isnan(A)))
            fprintf('File %s has NaN elements in Spec Feats\n',str)
            break
        end
        
        if sum(sum(isinf(A)))
            fprintf('File %s has Inf elements in Spec Feats\n',str)
            break
        end
        
        % cat data
        data = ([data;single(T)]);
        targets = ([targets;single(A)]);
        clv = [clv size(T,1)];
    end
    
    
    if sum(sum(isnan(T))) || sum(sum(isinf(T))) || sum(sum(isnan(A))) || sum(sum(isinf(A)))
        fprintf('File %s has NaN/Inf elements in feats \n',str)
        break
    end
    
    
    if sum(sum(isnan(data)))
        fprintf('There are NaN elements in data feats ... pausing ...\n')
        pause
    end
    
    if sum(sum(isnan(targets)))
        fprintf('There are NaN elements in target feats ... pausing ...\n')
        pause
    end
    
    save(strcat(matpath,'train',num2str(nb),'.mat'),'data','targets','clv','-v7.3');
end

