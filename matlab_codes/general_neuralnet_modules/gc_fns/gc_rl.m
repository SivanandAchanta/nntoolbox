function [p] = gc_rnn(p,gcth)
[p.gWi]  = gc(p.gWi,gcth);
[p.gWfr]  = gc(p.gWfr,gcth);
