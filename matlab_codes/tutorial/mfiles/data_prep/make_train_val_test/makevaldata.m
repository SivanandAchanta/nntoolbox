% Purpose : make validation data


for i = 1:length(files)
    
    [str,tok] = strtok(files(i).name,'.');
    fprintf('processing utt %s for val data \n',str);
    
    T = dlmread(strcat(text_valdir,files(i).name));
    A = dlmread(strcat(audio_valdir,str,aext));
    
    [tN,td] = size(T);
    [aN1,ad1] = size(A);
    
    nframes = min([tN aN1]);
    
    T = T(1:nframes,:);
    A = A(1:nframes,:);
    
    
    if sum(sum(isnan(T)))
        fprintf('File %s has NaN elements in Text Feats\n',str)
    end
    
    if sum(sum(isnan(A)))
        fprintf('File %s has NaN elements in Spec Feats\n',str)
    end
    
    if sum(sum(isinf(A)))
        fprintf('File %s has Inf elements in Spec Feats\n',str)
    end
    
    % cat data
    data = single([data;single(T)]);
    targets = single([targets;single(A)]);
    clv = [clv size(T,1)];
end

if sum(sum(isnan(data)))
    fprintf('There are NaN elements in data Feats\n')
    pause
    
end

if sum(sum(isnan(targets)))
    fprintf('There are NaN elements in target Feats\n')
    pause
    
end

save(strcat(matpath,'val','.mat'),'data','targets','clv','-v7.3');

