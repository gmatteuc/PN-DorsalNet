function [M,reorderingPermutation] = reorder_rows_by_peak_position(M)

    % get the number of rows in the input matrix
    numRows = size(M, 1);

    % initialize an array to store the row indices and peak positions
    peakPositions = zeros(numRows, 2);

    % iterate over each row
    for i = 1:numRows
        % find the index of the peak in the current row
        [~, peakPosition] = max(M(i, :));

        % store the row index and peak position in the array
        peakPositions(i, :) = [i, peakPosition];
    end

    % sort the array by the peak positions in descending order
    peakPositions = sortrows(peakPositions, -2);

    % reorder the rows of the input matrix according to the sorted array
    M = M(peakPositions(:, 1), :);
    
    % get reordering permutation
    reorderingPermutation = peakPositions(:, 1);
    
end