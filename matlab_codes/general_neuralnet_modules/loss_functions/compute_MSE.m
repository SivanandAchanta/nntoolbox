function [avg_loss] = compute_MSE(P,T)

avg_loss = mean(sum((T - P).^2,2));

end

