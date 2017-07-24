function [hm] = fp_cl(X,U,K,C,nin,f,sl)

% K - filter width
% X - Input matrix (sl x nin)
% U - filter coefficients ( K x C*nin)
% C - number of channels/filters

pad_frame = zeros(1,nin);
ac = zeros(sl,C);

% find if K is even or odd and pad accordingly

if mod(K,2) == 0
    Kby2 = round(K/2);
    X = [repmat(pad_frame,Kby2,1);X;repmat(pad_frame,Kby2-1,1)]; % after padding size of X is (sl + K - 1) x nin 

    for i = 1:C % process filter by filter
        for j = (1 + Kby2):(sl + Kby2) % process by sliding window across signal
            ac(j-Kby2,i) = sum(sum(U(:,(i-1)*nin+1:i*nin).*X(j-Kby2:j+Kby2-1,:),2),1); % width = K filters
        end
    end
    
else
    Kby2m1 = round(((K+1)/2 ) - 1);
    X = [repmat(pad_frame,Kby2m1,1);X;repmat(pad_frame,Kby2m1,1)]; % after padding size of X is (sl + K - 1) x nin 
    
    for i = 1:C % process filter by filter
        for j = (1 + Kby2m1):(sl + Kby2m1) % process by sliding window across signal
            ac(j-Kby2m1,i) = sum(sum(U(:,(i-1)*nin+1:i*nin).*X(j-Kby2m1:j+Kby2m1,:),2),1); % width = K filters
        end
    end
    
end


hm = get_actf(f,ac);

end