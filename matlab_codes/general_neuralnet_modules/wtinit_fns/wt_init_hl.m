function [Gp] = wt_init_hl(p,gpu_flag,nout,nin,si,btbf,wtdir,wtinit_meth,sgd_type)

if nout ~= nin
    fprintf('For Highway Layers fan_out = fan_in ...!!!');
    return
end

p.W = zeros(nout,nin);
p.Wt = zeros(nout,nout);
p.b = zeros(1,nout);
p.bt = btbf*ones(1,nout);

switch wtinit_meth
    case 'ki'
        % Kishore sirs Initilization ( This is where it all began !!! :) )
        
        maxweight = 3/sqrt(nin);
        p.W = 2*maxweight*rand(nin,nout) - maxweight;
        maxweight = 3/sqrt(nout);
        p.Wt = 2*maxweight*rand(nout,nout) - maxweight;
        
    case 'si'
        % Sparse Initialization
        
        % "On the importance of initialization and momentum for Deep Learning," Sutskever and Martens
        scale_in = 1/nout;
        scale_out = 1/nl(3);
        scale_hid = 1/nout;
        
        p.W = scale_in*randn(nin,nout);
        
        Ws = scale_hid*randn(nout,nout);
        for j = 1:nout
            Ws(j,randperm(nout,nout-nnz)) = 0;
        end
        p.Wt = Ws;
        
        if set_specradflag
            p.Wt = p.Wt*(specrad/abs(eigs(p.Wt,1,'lm',opts)));
        end
        
    case 'gi'
        % Sparse Initialization
        
        % "A Simple Way to Initialize RNNs using ReLp.U,", Q.V.Le, N.Jaitly and GEH
        scale_in = 1/nout;
        scale_hid = 1/nout;
        
        p.W = si*randn(nin,nout);
        p.Wt = si*randn(nin,nout);
        
        
    case 'di'
        % Sparse Initialization
        
        % "A Simple Way to Initialize RNNs using ReLp.U,", Q.V.Le, N.Jaitly and GEH
        scale_in = 1/nout;
        scale_hid = 1/nout;
        
        p.W = si*randn(nin,nout);
        p.Wt = si*randn(nin,nout);
        
    case 'lw'
        % Load pre-stored weights ... (Note architecture name must be specified !!!)
        % arch_name = '247L250R78L_rnn_di_l20_lr0.0001_mf0.3_gc1_si0.01_ri0.1_so0.01_rnn_mvni_mvno_50'
        % arch_name = '247L500R78L_rnn_di_l20_lr0.001_mf0.3_gc1_si0.01_ri0.01_so0.01_rnn_mvni_mvno_34'
        % arch_name = '247L300N78L_rnn_di_l20_lr0.03_mf0.3_gc1_si0.01_ri0.1_so0.01_rnn_mvni_mvno_37'
        arch_name = '247L500E150L_rnn_di_l20_lr0.003_mf0.3_gc1_si0.01_ri0.1_so0.01_rnn_mvni_mvno_40'
        
        load(strcat(wtdir,'W_',arch_name,'.mat'))
        p.W = p.W';
        p.b = p.b';
        
        
    case 'yi'
        % Yoshua Initialization Scheme
        
        % Ref : "p.Understanding the difficulty of training deep feedforward neural networks," Glorot,Xavier; Bengio, Yohsua (arXiv 06/02/15)
        
        maxweight = sqrt(6/(nin+nout));
        p.W = 2*maxweight*rand(nin,nout) - maxweight;
        maxweight = sqrt(6/(nout+nout));
        p.Wt = 2*maxweight*rand(nout,nout) - maxweight;
        
        
    case 'ri'
        % ReLp.U Initialization Scheme
        
        % Ref : Delving Deep into Rectifiers (arXiv 06/02/15)
        maxweight = sqrt(2/nin);
        p.W = maxweight*randn(nin,nout);
        maxweight = sqrt(2/nout);
        p.Wt = maxweight*randn(nout,nout);
    case 'rw'
        % Ref : "Random Walk Initalization for training very deep FFNNs" and "A Simple Way to Initalize Recurrent Neural Nets"
        
        maxweight = 1/nout;
        if strcmp(f(1),'R')
            g = sqrt(2)*exp(1.2/(max([nout 6])-2.4));
        elseif strcmp(f(1),'L')
            g = exp(maxweight/2);
        else
            g = 1.2;
        end
        p.W = g*maxweight*randn(nin,nout);
        maxweight = 1/nout;
        p.Wt = g*maxweight*randn(nout,nout);
        maxweight = 1/nl(3);

    otherwise

        fprintf('Please enter any of the above initialization methods \n');
        return        
        
end

p.W = p.W';
p.Wt = p.Wt';
p.b = p.b';
p.bt = p.bt';

disp('size of weight matrices');
size(p.W)
size(p.b)
size(p.Wt)
size(p.bt)

switch sgd_type
    case 'sgdcm'
        if gpu_flag
            Gp.W = gpuArray(p.W);  Gp.Wt = gpuArray(p.Wt);  Gp.b = gpuArray(p.b);
            Gp.pdW = gpuArray(zeros(size(p.W)));  Gp.pdWt = gpuArray(zeros(size(p.Wt)));
            Gp.pdb = gpuArray(zeros(size(p.b)));
        else
            Gp.W = p.W;  Gp.Wt = p.Wt;  Gp.b = p.b; Gp.bt = p.bt;
            Gp.pdW = zeros(size(p.W));  Gp.pdWt = zeros(size(p.Wt));
            Gp.pdb = zeros(size(p.b)); Gp.pdbt = zeros(size(p.bt));
        end
        
        
        
    case 'adadelta'
        if gpu_flag
            Gp.W = gpuArray(p.W);  Gp.Wt = gpuArray(p.Wt);  Gp.b = gpuArray(p.b);
            Gp.pdW = gpuArray(zeros(size(p.W)));  Gp.pdWt = gpuArray(zeros(size(p.Wt)));
            Gp.pdb = gpuArray(zeros(size(p.b)));
            Gp.pmsgW = gpuArray(zeros(size(p.W)));  Gp.pmsgWt = gpuArray(zeros(size(p.Wt)));
            Gp.pmsgb = gpuArray(zeros(size(p.b)));
            Gp.pmsxW = gpuArray(zeros(size(p.W)));  Gp.pmsxWt = gpuArray(zeros(size(p.Wt)));
            Gp.pmsxb = gpuArray(zeros(size(p.b)));
        else
            Gp.W = p.W;  Gp.Wt = p.Wt;  Gp.b = p.b; Gp.bt = p.bt;
            Gp.pdW = zeros(size(p.W));  Gp.pdWt = zeros(size(p.Wt));
            Gp.pdb = zeros(size(p.b)); Gp.pdbt = zeros(size(p.bt));
            Gp.pmsgW = zeros(size(p.W));  Gp.pmsgWt = zeros(size(p.Wt));
            Gp.pmsgb = zeros(size(p.b)); Gp.pmsgbt = zeros(size(p.bt));
            Gp.pmsxW = zeros(size(p.W));  Gp.pmsxWt = zeros(size(p.Wt));
            Gp.pmsxb = zeros(size(p.b)); Gp.pmsxbt = zeros(size(p.bt));
        end
        
    case 'adam'
        if gpu_flag
            Gp.W = gpuArray(p.W);  Gp.Wt = gpuArray(p.Wt);  Gp.b = gpuArray(p.b);
            Gp.pmW = gpuArray(zeros(size(p.W)));  Gp.pmWt = gpuArray(zeros(size(p.Wt)));
            Gp.pmb = gpuArray(zeros(size(p.b)));
            Gp.pvW = gpuArray(zeros(size(p.W)));  Gp.pvWt = gpuArray(zeros(size(p.Wt)));
            Gp.pvb = gpuArray(zeros(size(p.b)));
        else
            Gp.W = p.W;  Gp.Wt = p.Wt;  Gp.b = p.b; Gp.bt = p.bt;
            Gp.pmW = zeros(size(p.W));  Gp.pmWt = zeros(size(p.Wt));
            Gp.pmb = zeros(size(p.b)); Gp.pmbt = zeros(size(p.bt));
            Gp.pvW = zeros(size(p.W));  Gp.pvWt = zeros(size(p.Wt));
            Gp.pvb = zeros(size(p.b)); Gp.pvbt = zeros(size(p.bt));
        end
        
end
