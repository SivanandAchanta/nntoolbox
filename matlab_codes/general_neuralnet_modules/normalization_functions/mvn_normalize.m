function[X] = mvn_normalize(X,mvnivec,m,v)
I1 = bsxfun(@minus,X(:,mvnivec),m(:,mvnivec));
I1 = bsxfun(@rdivide,I1,v(:,mvnivec)+1e-5);
X(:,mvnivec) = I1;
end