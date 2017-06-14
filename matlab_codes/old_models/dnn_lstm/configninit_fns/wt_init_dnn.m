
W = zeros(1,sum(nl_dnn(1:end-1).*nl_dnn(2:end)));
b = zeros(1,sum(nl_dnn(2:end)));
scale = 0.1;

switch wtinit_meth_dnn
    case 'ki'
        % Kishore sirs Initilization ( This is where it all began !!! :) )
        
        for i = 1:nh_dnn
            maxweight = 3/sqrt(nl_dnn(i));
            W(1,wtl_dnn(i):wtl_dnn(i+1)-1) = 2*maxweight*rand(1,nl_dnn(i)*nl_dnn(i+1)) - maxweight;
        end
        
    case 'si'
        % Sparse Initialization
        
        % "On the importance of initialization and momentum for Deep Learning," Sutskever and Martens
        
        for i = 1:nh_dnn
            Ws = scale*randn(nl_dnn(i+1),nl_dnn(i));
            nh_dnn = nl_dnn(i+1);
            for j = 1:nh_dnn
                Ws(j,randperm(nl_dnn(i),nl_dnn(i)-20)) = 0;
            end
            W(1,wtl_dnn(i):wtl_dnn(i+1)-1) = Ws(:)';
        end
        
    case 'lw'
        % Load pre-stored weights ... (Note architecture name must be specified !!!)
        
        load(strcat(wtdir,'W_',arch_name,'.mat'))
        
    case 'yi'
        % Yoshua Initialization Scheme
        
        % Ref : "Understanding the difficulty of training deep feedforward neural networks," Glorot,Xavier; Bengio, Yohsua (arXiv 06/02/15)
        
        for i = 1:nh_dnn
            maxweight = sqrt(6/(nl_dnn(i) + nl_dnn(i+1)));
            Wr = 2*maxweight*rand(nl_dnn(i+1),nl_dnn(i)) - maxweight;
            W(1,wtl_dnn(i):wtl_dnn(i+1)-1) = Wr(:)';
        end
        
    case 'ri'
        % ReLU Initialization Scheme
        
        % Ref : Delving Deep into Rectifiers (arXiv 06/02/15)
        
        for i = 1:nh_dnn
            maxweight = sqrt(2/nl_dnn(i));
            W(1,wtl_dnn(i):wtl_dnn(i+1)-1) = maxweight*randn(1,nl_dnn(i)*nl_dnn(i+1));
        end
        
end

disp('size of weight matrix');
size(W)


switch sgd_type
    case 'sgdcm'
        pdW = zeros(size(W));
        pdb = zeros(size(b));
        
    case 'adadelta'
        
        pdW = zeros(size(W));
        pdb = zeros(size(b));
        
        pmsgW = zeros(size(W));
        pmsgb = zeros(size(b));
        
        pmsxW = zeros(size(W));
        pmsxb = zeros(size(b));
        
    case 'adam'
        
        pmW = zeros(size(W));
        pmb = zeros(size(b));
        
        pvW = zeros(size(W));
        pvb = zeros(size(b));
        
        
end

