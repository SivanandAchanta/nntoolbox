% Purpose : Gradients for LSTM parameters using error signals
% Ref : LSTM : A Search Space Odyssey (Greff et.al arxiv. 2014)
% Vanilla LSTM

function [gWz,gRz,gbz,gWf,gRf,gbf,gWc,gRc,gbc] = gradients_gru(X,hm,fm,dfm,dzm,dcm,nout)

hpm = [zeros(1,nout);hm(1:end-1,:)];

gWz = dzm'*X;
gWf = dfm'*X;
gWc = dcm'*X;

gRz = dzm'*hpm;
gRf = dfm'*hpm;
gRc = (fm.*dcm)'*(hpm);

gbz = sum(dzm,1)';
gbf = sum(dfm,1)';
gbc = sum(dcm,1)';

end

