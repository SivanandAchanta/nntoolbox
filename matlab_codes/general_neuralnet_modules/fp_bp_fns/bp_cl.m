function [p,Ebb] = bp_cl(p,Eb,ym,X,K,nin,nout,f,sl)


der_f = get_derf(nout,f,ym,sl);
Eb = der_f.*Eb;

pad_frame = zeros(1,nin);
Ebb = zeros(sl+K-1,nin);
gU = zeros(size(p.U));

% find if K is even or odd and pad accordingly

if mod(K,2) == 0
    Kby2 = round(K/2);
    X = [repmat(pad_frame,Kby2,1);X;repmat(pad_frame,Kby2-1,1)]; % after padding size of X is (sl + K - 1) x nin
    
    for i = 1:nout % process filter by filter
        %         Ebc = Eb(:,i);
        %         Ebcm = repmat(Ebc,1,nin);
        %         Ebcm = [repmat(pad_frame,Kby2,1);Ebcm;repmat(pad_frame,Kby2-1,1)]; % after padding size of X is (sl + K - 1) x nin
        
        for j = (1 + Kby2):(sl + Kby2) % process by sliding window across signal
            Ebb(j-Kby2:j+Kby2-1,:) = Ebb(j-Kby2:j+Kby2-1,:) + p.U(:,(i-1)*nin+1:i*nin)*Eb(j-Kby2m1,i); % width = K filters
            gU(:,(i-1)*nin+1:i*nin) = gU(:,(i-1)*nin+1:i*nin) + X(j-Kby2:j+Kby2-1,:)*Eb(j-Kby2,i);
        end
    end
    Ebb = Ebb((1 + Kby2):(sl + Kby2-1),:);
else
    Kby2m1 = round(((K+1)/2 ) - 1);
    X = [repmat(pad_frame,Kby2m1,1);X;repmat(pad_frame,Kby2m1,1)]; % after padding size of X is (sl + K - 1) x nin
    
    for i = 1:nout % process filter by filter
        %         Ebc = Eb(:,i);
        %         Ebcm = repmat(Ebc,1,nin);
        %         Ebcm = [repmat(pad_frame,Kby2m1,1);Ebcm;repmat(pad_frame,Kby2m1,1)]; % after padding size of X is (sl + K - 1) x nin
        
        for j = (1 + Kby2m1):(sl + Kby2m1) % process by sliding window across signal
            
            Ebb(j-Kby2m1:j+Kby2m1,:) = Ebb(j-Kby2m1:j+Kby2m1,:) + p.U(:,(i-1)*nin+1:i*nin)*Eb(j-Kby2m1,i); % width = K filters
            gU(:,(i-1)*nin+1:i*nin) = gU(:,(i-1)*nin+1:i*nin) + X(j-Kby2m1:j+Kby2m1,:)*Eb(j-Kby2m1,i);
        end
    end
    Ebb = Ebb((1 + Kby2m1):(sl + Kby2m1),:);
end

p.gU = gU;

end
