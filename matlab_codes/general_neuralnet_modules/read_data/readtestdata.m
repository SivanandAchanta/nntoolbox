load(strcat(datadir,testdataname));
test_batchdata = single(data);
if intmvnf
    I1 = bsxfun(@minus,test_batchdata(:,mvnivec),m(:,mvnivec));
    I1 = bsxfun(@rdivide,I1,v(:,mvnivec)+1e-5);
    test_batchdata(:,mvnivec) = I1;
    clear I1
end

if intmmnf
    I1 = bsxfun(@minus,test_batchdata(:,mvnivec),mini(:,mvnivec));
    I1 = bsxfun(@rdivide,I1,maxi(:,mvnivec)+1e-5);
    test_batchdata(:,mvnivec) = I1;
    clear I1
end


test_batchdata = test_batchdata(:,invec);

test_batchtargets = single(targets);
if outtmvnf
    I1 = bsxfun(@minus,test_batchtargets,mo);
    I1 = bsxfun(@rdivide,I1,vo+1e-5);
    test_batchtargets = I1;
    clear I1
end
if outtmmnf
    I1 = bsxfun(@minus,test_batchtargets,minv);
    I1 = bsxfun(@rdivide,I1,maxv);
    test_batchtargets = I1;
    clear I1
end

test_batchtargets = single(test_batchtargets(:,outvec));
clv = clv(:)';
test_clv = cumsum([1 clv]);
test_numbats = length(test_clv) - 1;
clear data targets clv

if sum(sum(isnan(test_batchdata)))
    disp('there are NaN eles in test data');
    pause
end

if sum(sum(isnan(test_batchtargets)))
    disp('there are NaN eles in test targets');
    pause
end

din = length(invec);
dout = length(outvec);
