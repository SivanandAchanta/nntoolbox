function [gW] = gc(gW,gcth)

lg = sqrt(sum(gW.^2,2)); ix_int = find(lg > gcth);
if ~isempty(ix_int); gW(ix_int,:) = gcth*bsxfun(@rdivide,gW(ix_int,:),lg(ix_int)); end;


end