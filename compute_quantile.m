function quantileVal = compute_quantile(vector, scalar)
% sort the vector
vector = sort(vector);
% find the percentile rank of the scalar
scalarRank = sum(vector <= scalar) / length(vector);
% compute the quantile
quantileVal = scalarRank;
end

