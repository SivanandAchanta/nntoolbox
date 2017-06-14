
W = zeros(1,sum(nl(1:end-1).*nl(2:end)));
b = zeros(1,sum(nl(2:end)));
scale = 0.1;

switch wtinit_meth
    case 'ki'
        % Kishore sirs Initilization ( This is where it all began !!! :) )
        
        for i = 1:nh
            maxweight = 3/sqrt(nl(i));
            W(1,wtl(i):wtl(i+1)-1) = 2*maxweight*rand(1,nl(i)*nl(i+1)) - maxweight;
        end
        
    case 'si'
        % Sparse Initialization
        
        % "On the importance of initialization and momentum for Deep Learning," Sutskever and Martens
        
        for i = 1:nh
            Ws = scale*randn(nl(i+1),nl(i));
            Nh = nl(i+1);
            for j = 1:Nh
                Ws(j,randperm(nl(i),nl(i)-20)) = 0;
            end
            W(1,wtl(i):wtl(i+1)-1) = Ws(:)';
        end
        
    case 'lw'
        % Load pre-stored weights ... (Note architecture name must be specified !!!)
        
        load(strcat(wtdir,'W_',arch_name,'.mat'))
        
    case 'yi'
        % Yoshua Initialization Scheme
        
        % Ref : "Understanding the difficulty of training deep feedforward neural networks," Glorot,Xavier; Bengio, Yohsua (arXiv 06/02/15)
        
        for i = 1:nh
            maxweight = sqrt(6/(nl(i) + nl(i+1)));
            Wr = 2*maxweight*rand(nl(i+1),nl(i)) - maxweight;
            W(1,wtl(i):wtl(i+1)-1) = Wr(:)';
        end
        
    case 'ri'
        % ReLU Initialization Scheme
        
        % Ref : Delving Deep into Rectifiers (arXiv 06/02/15)
        
        for i = 1:nh
            maxweight = sqrt(2/nl(i));
            W(1,wtl(i):wtl(i+1)-1) = maxweight*randn(1,nl(i)*nl(i+1));
        end
        
end

disp('size of weight matrix');
size(W)

if gpu_flag
    GW = gpuArray(W);
    Gb = gpuArray(b);
else
    GW = W;
    Gb = b;
end

switch sgd_type
    case 'sgdcm'
        pdW = zeros(size(W));
        pdb = zeros(size(b));
        
        if gpu_flag
            GpdW = gpuArray(pdW);
            Gpdb = gpuArray(pdb);
        else
            GpdW = (pdW);
            Gpdb = (pdb);
        end
        
    case 'adadelta'
        pdW = zeros(size(W));
        pdb = zeros(size(b));
        
        pmsgW = zeros(size(W));
        pmsgb = zeros(size(b));
        
        pmsxW = zeros(size(W));
        pmsxb = zeros(size(b));
        
        if gpu_flag
            GpdW = gpuArray(pdW);
            Gpdb = gpuArray(pdb);
            GpmsgW = gpuArray(pdW);
            Gpmsgb = gpuArray(pdb);
            GpmsxW = gpuArray(pdW);
            Gpmsxb = gpuArray(pdb);
         
        else
            GpdW = (pdW);
            Gpdb = (pdb);
            GpmsgW = (pdW);
            Gpmsgb = (pdb);
            GpmsxW = (pdW);
            Gpmsxb = (pdb);
        end
        
    case 'adam'
        pmW = zeros(size(W));
        pmb = zeros(size(b));
        
        pvW = zeros(size(W));
        pvb = zeros(size(b));
        
        if gpu_flag
           GpmW = gpuArray(pmW);
           Gpmb = gpuArray(pmb);
           GpvW = gpuArray(pmW);
           Gpvb = gpuArray(pmb);
        else
           GpmW = pmW;
           Gpmb = pmb;
           GpvW = pmW;
           Gpvb = pmb;
        end
        
end
