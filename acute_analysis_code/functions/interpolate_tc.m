function [output_vector_orig,output_vector,...
    input_vector_orig,input_vector,...
    x_vector_orig,x_vector] = interpolate_tc(tun_vector,angle_step_orig,angle_step_interp,smoothfrac)

% get input and interpolate it to upsample
input_vector_orig=tun_vector;
x_vector_orig=0:angle_step_orig:(360-angle_step_orig);
x_vector=0:angle_step_interp:(360-angle_step_orig);
input_vector=interp1(x_vector_orig,input_vector_orig,x_vector,'spline');
% pad upsampled input to make the smoothing "circular"
starting_input_vector_length=length(input_vector);
padded_input_vector=[fliplr(input_vector),input_vector,input_vector];
% smooth padded vector
padded_input_vector_smoothed=smooth(padded_input_vector,smoothfrac,'loess');
% get rid of padding
output_vector=padded_input_vector_smoothed(...
    (starting_input_vector_length+1):...
    (2*starting_input_vector_length+1));
% downsampled smoothed vector
output_vector_orig=output_vector(1:(angle_step_orig./angle_step_interp):end); %#ok<BDSCA>

end

