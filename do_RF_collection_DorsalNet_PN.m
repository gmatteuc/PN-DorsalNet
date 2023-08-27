clear all %#ok<CLALL>
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

% set if to perform gabor fitting analysis
performgaborfittingbool=1;

% get list of STA layer folders
STA_layer_folder_list=dir(fullfile(resultpath,'*STA_results*'));
% get layer names
layer_names=cell(1,numel(STA_layer_folder_list));
for folder_idx=1:numel(STA_layer_folder_list)
    layer_names{folder_idx}=strrep(STA_layer_folder_list(folder_idx).name,'STA_results_','');
end

% set pars
pars = set_pars_PN();
DIR=pars.stimPars.DIR;
SR=1/pars.stimPars.frame_duration;
STAmaxlag = pars.STA_depth;
target_STAsize=[32 32];
STAwidth = target_STAsize(2); %pars.STA_width;
STAheight = target_STAsize(1); %pars.STA_height;
interpolation_factor=pars.interp_factor;

% loop over layers
for current_layer_id=5:numel(layer_names)%1:numel(layer_names)
    
    % get current folderpath
    current_folderpath=[STA_layer_folder_list(current_layer_id).folder,filesep,STA_layer_folder_list(current_layer_id).name];
    
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    
    % get list of STA files in current STA layer folders
    current_STA_file_list=dir(fullfile(current_folderpath,'*STA_results_neuron*'));
    % get original neuron number
    neuron_ids=NaN(1,numel(current_STA_file_list));
    for neuron_id=1:numel(current_STA_file_list)
        neuron_ids(neuron_id)=str2num(strrep(strrep(current_STA_file_list(neuron_id).name,'STA_results_neuron_',''),'.mat','')); %#ok<ST2NM>
    end
    % sort files by original neuron number
    [~,id_permutation] = sort(neuron_ids);
    current_STA_file_list=current_STA_file_list(id_permutation);
    
    % initialize common STA storage variables
    rsta=zeros(STAheight,STAwidth,STAmaxlag,numel(current_STA_file_list));
    wsta=zeros(STAheight,STAwidth,STAmaxlag,numel(current_STA_file_list));
    Zrsta=zeros(STAheight,STAwidth,STAmaxlag,numel(current_STA_file_list));
    Zwsta=zeros(STAheight,STAwidth,STAmaxlag,numel(current_STA_file_list));
    neuron_number=zeros(1,numel(current_STA_file_list));
    totalspikes=zeros(1,numel(current_STA_file_list));
    power=zeros(STAmaxlag,numel(current_STA_file_list));
    bestfr_power=zeros(1,numel(current_STA_file_list));
    relative_areas=zeros(STAmaxlag,numel(current_STA_file_list));
    lobe_numbers=zeros(STAmaxlag,numel(current_STA_file_list));
    contrast_factors=zeros(STAmaxlag,numel(current_STA_file_list));
    bestfr_lobe_numbers=zeros(1,numel(current_STA_file_list));
    bestfr_idx=zeros(1,numel(current_STA_file_list));
    Zwsta_fit_r2=zeros(STAmaxlag,numel(current_STA_file_list));
    Zwsta_fitted=zeros(STAheight,STAwidth,STAmaxlag,numel(current_STA_file_list));
    
    % loop over neurons
    for current_n=1:numel(current_STA_file_list)
        
        % load current neuron STA data
        current_filename=[current_STA_file_list(current_n).folder,filesep,current_STA_file_list(current_n).name];
        STAdata=load(current_filename);
        % verify neuron number correspondence
        assert(STAdata.Dneuronum==current_n)
        
        % store current STA data in common STA storage
        rsta(:,:,:,current_n)=STAdata.Dstafr;
        wsta(:,:,:,current_n)=STAdata.Dwstafr;
        Zrsta(:,:,:,current_n)=STAdata.DZstafr;
        Zwsta(:,:,:,current_n)=STAdata.DZwstafr;
        neuron_number(current_n)=STAdata.Dneuronum;
        totalspikes(current_n)=STAdata.Dtotspikes;
        
        %% STA power ------------------------------------------------------
        
        % get RF power for every frame
        for ff=1:STAmaxlag
            % select STA frame and interpolate it
            inputfr=Zwsta(:,:,ff,current_n);
            % set interpolation factr
            interp_factor=interpolation_factor;
            % interpolate
            outputfr = interpolate_RF_frame( inputfr, interp_factor );
            % compute STA power
            currfr_pow=smoothn(outputfr.^2,10000);
            power(ff,current_n)=mean(currfr_pow(:));
        end
        % select power best frames
        [maxpo,bestfr_power(current_n)]=max(power(:,current_n));
        
        %% STA shape features ------------------------------------------------------
        
        % set area limit to consider for lobe count
        input_pars.lobecount_relareath=0.05;
        % set binarization thereshold to consider for lobe count
        input_pars.bin_zth=3.5;
        % get RF shape features for every frame
        for ff=1:STAmaxlag
            % get and resize current frame
            current_Zrf_frame=squeeze(Zwsta(:,:,ff,current_n));
            current_Zrf_frame_resized=imresize(current_Zrf_frame,10,'bicubic');
            % set cropping function input
            crop_pixel_size=100;
            input_fr=current_Zrf_frame_resized;
            % crop current STA frame
            [ ~, crop_ridx, crop_cidx  ] = apply_crop( input_fr , crop_pixel_size ,0 );
            cropped_frame=input_fr(crop_ridx, crop_cidx);
            % compute STA contrast
            [ ~, ~, ~, contrast_factors(ff,current_n),...
                relative_areas(ff,current_n),...
                lobe_numbers(ff,current_n), ~  ] = ...
                get_shape_params_bis( cropped_frame,input_pars, 0 );
        end
        % get contrast factor best frame
        [~,best_idx]=max(contrast_factors(:,current_n),[],1);
        % store best frame idx and lobe count at best frame
        bestfr_lobe_numbers(current_n)=lobe_numbers(best_idx,current_n);
        bestfr_idx(current_n)=best_idx;
        
        %% STA Gabor fitting ------------------------------------------------------
        
        if performgaborfittingbool
            % decide if to plot rf fit diagnostics
            plotrffitdiagnosticsbool=0;
            % set options for gabor fitting
            gfitoptions.shape = 'elliptical';
            gfitoptions.runs  = 5;
            gfitoptions.parallel = true;
            gfitoptions.visualize = false;
            % loop over frames
            for ff=1:4%STAmaxlag
                % perform Gabor fitting
                current_rf_frame=Zwsta(:,:,ff,current_n);
                current_rf_frame_resized=imresize(current_rf_frame,2,'bicubic');
                current_rf_frame_resized_fitted = fit2dGabor(current_rf_frame_resized,gfitoptions);
                current_rf_frame_fitted=imresize(current_rf_frame_resized_fitted.patch,1/2,'bicubic');
                Zwsta_fitted(:,:,ff,current_n)=current_rf_frame_fitted;
                Zwsta_fit_r2(ff,current_n)=current_rf_frame_resized_fitted.r2;
            end
            if plotrffitdiagnosticsbool
                % visualize rf gabor fit diagnostics
                fv=figure('units','normalized','outerposition',[0 0 1 1]);
                v2 = VideoWriter([resultpath,filesep,current_layer_name,'_n',num2str(current_n)],'MPEG-4'); %#ok<TNMLP>            v2.FrameRate = 5;
                v2.FrameRate = 5;
                open(v2);
                for ff=1:STAmaxlag
                    % get current frame to plot
                    curr_fr_to_show=STAmaxlag-ff+1;
                    % get max and min of colorscale
                    temppixlv=squeeze(Zwsta(:,:,:,current_n));
                    caxmax=quantile(temppixlv(:),0.99);
                    caxmin=quantile(temppixlv(:),0.01);
                    % plot sequance of frames
                    subplot(1,2,1)
                    imagesc(squeeze(Zwsta(:,:,curr_fr_to_show,current_n))); colorbar; colormap(gray); caxis([caxmin,caxmax]);
                    title('raw RF')
                    subplot(1,2,2)
                    imagesc(squeeze(Zwsta_fitted(:,:,curr_fr_to_show,current_n))); colorbar; colormap(gray); caxis([caxmin,caxmax]);
                    title('gabor fitted RF')
                    suptitle([current_layer_name,' #',num2str(current_n),' - STA frame ',...
                        num2str(curr_fr_to_show),' - r2=  ',num2str(Zwsta_fit_r2(curr_fr_to_show,current_n)),...
                        ' - gosta = ',num2str(contrast_factors(bestfr_idx(current_n),current_n))])
                    % append frame to video
                    frame = getframe(gcf);
                    writeVideo(v2,frame)
                end
                % close video file
                close(v2);
                close all
            end
        end
        
        % output state message (neuron)
        message=['\nNeuron ',num2str(current_n),' (',current_layer_name,') added to RF data structure\n'];
        fprintf(message)
        close all
        
    end
    
    % save common STA storage for current layer
    save([resultpath,filesep,'RFs_datastructure_',current_layer_name],'rsta','wsta','Zrsta','Zwsta','neuron_number',...
        'totalspikes','power','bestfr_power','relative_areas','lobe_numbers','contrast_factors','Zwsta_fitted','Zwsta_fit_r2')
    
    % output state message (layer)
    message=['\n-----',current_layer_name,' RFs collected -----\n'];
    fprintf(message)
    
end