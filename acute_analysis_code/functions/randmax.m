function [ maxval, maxidx ] = randmax( input )

% [ max, maxidx ] = robustmax( input )
% compute maximum always outputing a scalar randomly choosing one peak in
% case of equipeaked input

maxval = max(input);
maxidx = find(input==max(input));
if length(maxidx)>1
    id=randperm(length(maxidx));
    id=id(1);
else
    id=1;
end

maxidx=maxidx(id);

end

