clear all
close all
clc

%% load noise data -------------------------------------------------

% set paths
codepath='E:\Backups\Personal_bk\DorsalNet\code';
oldcode='E:\Backups\Personal_bk\DorsalNet\acute_analysis_code';
datapath='E:\Backups\Personal_bk\DorsalNet\activations';
resultpath='E:\Backups\Personal_bk\DorsalNet\results';
stimulipath='E:\Backups\Personal_bk\DorsalNet\stimuli';
% add paths
addpath E:\Backups\Personal_bk\DorsalNet
addpath(genpath(codepath));
addpath(genpath(oldcode));
addpath(datapath)
addpath(resultpath)

% get list of noise response files
noise_file_list=dir(fullfile(datapath,'*Noise*'));
% get original noisemovie id number
noisemovie_ids=NaN(1,numel(noise_file_list));
for noise_file_id=1:numel(noise_file_list)
   noisemovie_ids(noise_file_id)=str2num(strrep(strrep(noise_file_list(noise_file_id).name,'Noise_responses',''),'_Matlab2.mat',''))+1; %#ok<ST2NM>
end
% sort filenames by original noisemovie id number
[~,id_permutation] = sort(noisemovie_ids);
noise_file_list=noise_file_list(id_permutation);

% loop over noise response files
for noise_file_id=1:numel(noise_file_list)
    tic
    % load grating activations
    Noise_activations_temp=load([noise_file_list(noise_file_id).folder,filesep,noise_file_list(noise_file_id).name]);
    if noise_file_id==1
        % initialize storage variable as the first loaded
        Noise_activations=Noise_activations_temp;
        % get layer names
        layer_names=fieldnames(Noise_activations);
    else
        % loop over layers to stack data in storage variable
        for current_layer_id=1:numel(layer_names)
            % get current layer name
            current_layer_name=layer_names{current_layer_id};
            % stack results
            Noise_activations.(current_layer_name)=cat(1,...
                Noise_activations.(current_layer_name),...
                Noise_activations_temp.(current_layer_name));
        end
    end
    toc
end

% get layer names
layer_names=fieldnames(Noise_activations);

% get stimulus types
stimulus_types={'noise'};

% get grating and plaid length
noise_length=size(Noise_activations.(layer_names{1}),3);

% set pars
pars = set_pars_PN();
SR=1/pars.stimPars.frame_duration;

%% perform STA analysis -------------------------------------------------

% get noise movie path
noise_path=[stimulipath,filesep,'Stimuli_DorsalNet_Matlab2.mat'];

% loop over layers
for current_layer_id=1:numel(layer_names)
    
    tic
    
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    
    % create STA output folder
    STA_output_folder=[resultpath,filesep,'STA_results_',current_layer_name];
    if not(exist(STA_output_folder)) %#ok<EXIST>
        mkdir(STA_output_folder)
    end
    
    % get activations for current layer (rectified and regularized)
    temp_activations=double(Noise_activations.(current_layer_name));
    activation_th=0; % quantile(temp_activations(:),0.05);
    temp_activations(temp_activations<=activation_th)=0;
    current_psthmat=max(temp_activations,...
        zeros(size(temp_activations)))...
        +eps*rand(size(temp_activations));
    % set STA anlaysis input
    inputpars.output_folder=STA_output_folder;
    inputpars.noise_path=noise_path;
    inputpars.current_layer_name=current_layer_name;
    % run STA analysis
    compute_STA_DorsalNet_activations_PN(current_psthmat,inputpars);
    
    toc
    
end