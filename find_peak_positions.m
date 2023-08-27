function peak_positions = find_peak_positions(M)

    % get the number of rows in the input matrix
    numRows = size(M, 1);
    % initialize an array to store the peak positions
    peak_positions = zeros(numRows, 2);
    % iterate over each row
    for i = 1:numRows
        % split the row into two halves
        firstHalf = M(i, 1:ceil(end/2));
        secondHalf = M(i, ceil(end/2)+1:end);
        % find the position of the peak in each half
        [~, peakPositionFirstHalf] = max(firstHalf);
        [~, peakPositionSecondHalf] = max(secondHalf);
        % store the peak positions in the array
        peak_positions(i, :) = [peakPositionFirstHalf, peakPositionSecondHalf + ceil(size(M, 2) / 2)];
    end
end
