function [gW,gb] = bpav_cpu(X,Y,ol,GW,otl,btl,wtl,fl,bs,nl,nh,cfn,l1,l2)

% gradients of weights of top most layer
ol_m = reshape(ol(1,otl(end-1):otl(end)-1),bs,nl(end));
ol_pl_m = reshape(ol(1,otl(end-2):otl(end-1)-1),bs,nl(end-1));

[der_f] = get_derf(nl(end),fl(end),ol_m,bs);

switch cfn
    case 'ls'
        costder = -(Y - ol_m)/bs;
        del_bp = costder.*der_f;
    case  'nll'
        %         costder = -(Out./ol_m);
        del_bp  = -(Y - ol_m)/bs;
end

own = reshape(GW(1,wtl(nh):wtl(nh+1)-1),nl(nh+1),nl(nh))';
gbv = sum(del_bp,1);
gWm = ((ol_pl_m'*del_bp) + l1*sign(own) + l2*2*own);

gb = (zeros(1,btl(end-1)));
gb(1,btl(nh):btl(end)-1) = gbv;
gW = (zeros(1,wtl(end)-1));
gW(1,wtl(nh):wtl(nh+1)-1) = reshape(gWm',1,numel(gWm));

% gradients of weights of inner hidden layers
for j = nh-1:-1:1
    
    ol_m = reshape(ol(1,otl(j):otl(j+1)-1),bs,nl(j+1));
    if (j-1) ~=0
        ol_pl_m = reshape(ol(1,otl(j-1):otl(j)-1),bs,nl(j));
    else
        ol_pl_m = X;
    end
       
    [der_f] = get_derf(nl(j+1),fl(j),ol_m,bs);
    
    wdel_bp = del_bp*(own'); % weghted del_bps
    del_bp = wdel_bp.*der_f;
    gbv = sum(del_bp,1);
    own = reshape(GW(1,wtl(j):wtl(j+1)-1),nl(j+1),nl(j))'; % only weights are considered , ingore the first row which are biases
    gWm = ((ol_pl_m'*del_bp) + l1*sign(own) + l2*2*own);
    
    gb(1,btl(j):btl(j+1)-1) = gbv;
    gW(1,wtl(j):wtl(j+1)-1) = reshape(gWm',1,numel(gWm));
    
end