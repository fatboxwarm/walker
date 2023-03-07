function X = reverseMeanNorm(normalized_X, mean_X, range_X)
%reverseMeanNorm reverse normalized_X into X using mean_X & range_X
%   use meanNorm(X) to normalize X
X=normalized_X*range_X+mean_X;
end