function concatenated_datstructure = concatenate_layers(layer_datstructure)
% get layers list
layer_names=fieldnames(layer_datstructure);
% initialize concatenated datstructure
concatenated_datstructure=[];
% loop over layers
for layer_id=1:numel(layer_names)
    % get current layer
    layer_name=layer_names{layer_id};
    % concatenate data
    concatenated_datstructure=cat(1,concatenated_datstructure,layer_datstructure.(layer_name));
end
end

