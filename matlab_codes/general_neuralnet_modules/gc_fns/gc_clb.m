function [p] = gc_clb(p,gcth,K_conv)

if K_conv == 3
    [p.gU1]  = gc(p.U1,gcth);
    [p.gU2] = gc(p.U2,gcth);
    [p.gU3] = gc(p.U3,gcth);
elseif K_conv == 8
    [p.gU1]  = gc(p.U1,gcth);
    [p.gU2] = gc(p.U2,gcth);
    [p.gU3] = gc(p.U3,gcth);
    [p.gU4]  = gc(p.U4,gcth);
    [p.gU5] = gc(p.U5,gcth);
    [p.gU6] = gc(p.U6,gcth);
    [p.gU7]  = gc(p.U7,gcth);
    [p.gU8] = gc(p.U8,gcth);
    
end