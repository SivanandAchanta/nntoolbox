function [GW, Gb] = weightoperations_mnc(GW, Gb, mnc)

for j = 1:nh
    rWt = reshape(GW(1,wtl(j):wtl(j+1)-1),nl(j+1),(nl(j)))';
    lenwv{j} = sqrt(sum(rWt(2:end,:).^2)'); % weight vector lengths excluding biases
    idxin = find(lenwv{j} > mnc);
    
    if ~isempty(idxin)
        rWt(2:end,idxin) = bsxfun(@rdivide,rWt(2:end,idxin),(lenwv{j}(idxin))'/mnc);
    end
    %                 mlenwv{j} = sum(rWt(2:end,:).^2)'; % modified weight vector lengths excluding biases
    
    % replace the original weights with the modified weights
    GW(1,wtl(j):wtl(j+1)-1) = reshape(rWt',1,(nl(j))*nl(j+1));
end

end
