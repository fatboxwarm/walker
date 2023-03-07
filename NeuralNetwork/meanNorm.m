function [normalized_X, mean_X, range_X] = meanNorm(X)
%meanNorm returns the normalized col vec X and its mean/range
%   use reverseMeanNorm to put X back together
range_X=max(X)-min(X);
mean_X=mean(X);
normalized_X=(X-mean_X)/range_X;
end