function output_vector = ensure_is_colum(input_vector)

if size(input_vector,1)>=size(input_vector,2) % column
    output_vector=input_vector;
elseif size(input_vector,1)<size(input_vector,2) %row
    output_vector=input_vector';
end

end

