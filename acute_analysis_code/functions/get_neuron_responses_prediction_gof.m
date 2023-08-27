function [explvar_tuning,rmse_tuning,explvar_dynamics,rmse_dynamics] = get_neuron_responses_prediction_gof(obs_fr,obs_count,pred_fr,pred_count,target_dir_idx)

% compare observed and predicted dynamics
explvar_dynamics=corr(obs_fr(:,target_dir_idx),pred_fr(:,target_dir_idx)).^2;
rmse_dynamics=sqrt(nanmean((obs_fr(:,target_dir_idx)-pred_fr(:,target_dir_idx)).^2));

% compare observed and predicted tuning curve
explvar_tuning=(corr(obs_count',pred_count')).^2;
rmse_tuning=sqrt(nanmean((obs_count'-pred_count').^2));

end

