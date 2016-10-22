function [avg_loss] = compute_nmlMSE(P,T)

avg_loss = mean(sum((T - P).^2,2)./(sum(T.^2,2)));

end

