function modularity = compute_block_modularity_PN(input_RDM, blocklabels)

% NOTE: We can define the block structure modularity as the difference between the
% average within-block dissimilarity and the overall average dissimilarity.
% A positive modularity score indicates that the average within-block
% dissimilarity is less than the overall average, suggesting a strong block
% structure. Conversely, a negative or near-zero modularity score suggests a
% weak or non-existent block structure.

% compute the overall average dissimilarity
overall_avg = nanmean(input_RDM(:));
% unique blocks
blocks = unique(blocklabels);
% initialize sums for within-block averages
block_sums = 0;
block_counts = 0;
% For each block
for i = 1:length(blocks)
    % indices of this block
    block_indices = blocklabels == blocks(i);
    % compute the average dissimilarity within this block
    block_avg = nanmean(input_RDM(block_indices, block_indices), 'all');
    % update the sums for the within-block averages
    block_sums = block_sums + block_avg;
    block_counts = block_counts + 1;
end

% compute the average of the within-block averages
block_avg = block_sums / block_counts;
% cmpute the block structure modularity
modularity = (1-block_avg) - (1-overall_avg);

end