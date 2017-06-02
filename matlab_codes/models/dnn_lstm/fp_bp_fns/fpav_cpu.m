function [ol] = fpav_cpu(X,Wt,b,nl,fl,nh,wtl,btl,Nin)
% fp - forward propogation of input through the network

% Input : (1) X    - Input data matrix (N x din)
%         (2) Wt   - Weight cell (Weights at all layers including biases)
%         (3) nl   - Num of nodes at each layer (1 x (nh+1) )
%         (4) fl   - Function at each layer ('N' or 'S' or 'L')
%         (5) nh   - number of hidden layers
%       (6,7) a,b  - parmas of tanh
%         (8) wtl  - number of weights in each layer
%         (9) berp - bernoulli probability of neuron being alive
% %      (10) mnc  - max-norm constraint on the weight vector
%        (11) Nin  - Number of input cases

% Output : (1) ol - Output cell (outputs at each layer)
%          (2) ac - Activation cell (activation values at each layer)


% compute o/p from 1st layer

Xm = [ones(Nin,1) X]; % ( Nin x (din+1))
rWt = reshape(Wt(1,wtl(1):wtl(2)-1),nl(2),nl(1))';

pac = Xm*[b(1,btl(1):btl(2)-1);rWt];
ac = get_actf(fl(1),pac);
ol = reshape(ac,1,Nin*nl(2)); % (1 x (Nin*nl(2)))

% compute o/p from other layers

for i = 2:nh
    
    Xm = [ones(Nin,1) ac]; % ( Nin x (din+1))
    rWt = reshape(Wt(1,wtl(i):wtl(i+1)-1),nl(i+1),nl(i))';
    
    pac = Xm*[b(1,btl(i):btl(i+1)-1);rWt];
    ac = get_actf(fl(i),pac);
    ol = [ol reshape(ac,1,Nin*nl(i+1))];
    
end


end

