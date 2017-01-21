function [X] = get_X(batch_data,li)

X = batch_data(:,:,li);
%X = X/max(abs(X));

end