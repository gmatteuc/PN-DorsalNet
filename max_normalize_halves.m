function M = max_normalize_halves(M)
    % get the number of rows in the input matrix
    numRows = size(M, 1);

    % iterate over each row
    for i = 1:numRows
        % split the row into two halves
        firstHalf = M(i, 1:ceil(end/2));
        secondHalf = M(i, ceil(end/2)+1:end);
%         % normalize each half by its maximum value
%         firstHalf = (firstHalf / max(firstHalf));
%         secondHalf = (secondHalf / max(secondHalf)) - 10.*eps;
        % normalize each half by its maximum value
        firstHalf = ( (firstHalf-min(firstHalf)) / ( max(firstHalf)-min(firstHalf) ) );
        secondHalf = ( (secondHalf-min(secondHalf)) / ( max(secondHalf)-min(secondHalf) ) ) - 10.*eps;

        % concatenate the normalized halves back together
        M(i, :) = [firstHalf, secondHalf];
    end
    
end
