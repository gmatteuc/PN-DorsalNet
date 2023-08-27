function smoothedMatrix = gaussianSmooth1D(inputMatrix, sigma, dim)

    % Check if the specified dimension is 1 or 2.
    if dim ~= 1 && dim ~= 2
        error('The specified dimension must be 1 (for rows) or 2 (for columns).')
    end

    % Create a Gaussian kernel.
    sizeKernel = 2*ceil(2*sigma) + 1; % This ensures the kernel size is an odd number.
    x = -(sizeKernel-1)/2 : (sizeKernel-1)/2;
    kernel = exp(-x.^2/(2*sigma^2)) / (sqrt(2*pi)*sigma);

    % Normalize the kernel to sum to 1.
    kernel = kernel / sum(kernel);

    % Apply the convolution.
    if dim == 1
        smoothedMatrix = conv2(1, kernel, inputMatrix, 'same');
    else
        smoothedMatrix = conv2(kernel, 1, inputMatrix, 'same');
    end

end
