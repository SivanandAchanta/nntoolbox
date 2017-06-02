function [p] = gc_gru(p,gcth)
[p.gWz] = gc(p.gWz,gcth);
[p.gRz] = gc(p.gRz,gcth);
[p.gWf] = gc(p.gWf,gcth);
[p.gRf] = gc(p.gRf,gcth);
[p.gWc] = gc(p.gWc,gcth);
[p.gRc] = gc(p.gRc,gcth);
