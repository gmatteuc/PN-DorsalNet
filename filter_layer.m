function boolfilt = filter_layer(layers_ids,target_layers_ids)

boolfilt=zeros(size(layers_ids));
for current_target_layers_id=target_layers_ids
    boolfilt=boolfilt+(layers_ids==current_target_layers_id);
end

end

