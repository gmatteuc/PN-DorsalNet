% ------------------------- OVERALL PATTERN VERSUS COMPONENT PREDICTION ANALYSIS (DN) -------------------------

clear all %#ok<CLALL>
close all
clc

%% load data -------------------------------------------------

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

% set layers to consider
target_layers_ids=[1:7];

% create prediction results folder
outputfolder=[resultpath,filesep,'STA_based_predictions'];
if not(exist(outputfolder)) %#ok<EXIST>
    mkdir(outputfolder)
end

% set pars
pars = set_pars_PN();
DIR=pars.stimPars.DIR;
SF='default';
TF='default';
SR=1/pars.stimPars.frame_duration;
STAmaxlag = pars.STA_depth;
target_STAsize=[32 32];
STAwidth = target_STAsize(2); %pars.STA_width;
STAheight = target_STAsize(1); %pars.STA_height;
STAborder = 5;
interpolation_factor=pars.interp_factor./4;
cell_types_codes=[1,2,0];
cell_types_names={'unclassified','component','pattern'};
stim_types={'gratings','plaids'};
frame_dur=pars.stimPars.frame_duration;
stim_dur=pars.stimPars.stim_time;
prediction_isi=pars.prediction_isi;

% load grating and plaid stimuli
load([stimulipath,filesep,'Stimuli_DorsalNet_Matlab2.mat']) % bis ---> everything as in exp but dirs = 12
% get rid of color channel dimension
Gratings=squeeze(nanmean(Gratings,2));
Plaids=squeeze(nanmean(Plaids,2));
% store stimuli
Stimuli{1}=Gratings;
Stimuli{2}=Plaids;
clear Gratings
clear Plaids

% load tuning anlysis results
load([resultpath,filesep,'Tuning_datastructure.mat'])

% get layer names
layer_names=fieldnames(Rlabel_observed);

% get stimulus duration bins
stim_dur_bins=size(PSTH_observed.(layer_names{1}),2);

% set wether to secompute predictions or not
recomputepredsbool=0;

%% compute STA based predictions -------------------------------------------------

if recomputepredsbool
    
    % loop over layers
    for current_layer_id=1:numel(layer_names)
        
        % get current layer name
        current_layer_name=layer_names{current_layer_id};
        
        % get number of neurons in current layer
        neu_n=size(Rlabel_observed.(current_layer_name),1);
        
        % refetch oberved index value vectors
        Zp_O=Zp_observed.(current_layer_name);
        Zc_O=Zc_observed.(current_layer_name);
        pDIR_O=pDIR_observed.(current_layer_name)(:,1);
        PI_O=Zp_O-Zc_O;
        ctlabel_O=Clabel_observed.(current_layer_name);
        obs_OSI=OSI_observed.(current_layer_name)(:,1);
        obs_DSI=DSI_observed.(current_layer_name)(:,1);
        obs_FR=permute(repmat(PSTH_observed.(current_layer_name),[1,1,1,length(DIR)]),[2,4,1,3]); % NB: add extraction of all PSTHs bfore
        obs_COUNT=permute(TC_observed.(current_layer_name),[2,1,3]);
        % initialize predicted index value vectors
        Zp_P=NaN(neu_n,1);
        Zc_P=NaN(neu_n,1);
        pDIR_P=NaN(neu_n,1);
        PI_P=NaN(neu_n,1);
        ctlabel_P=NaN(neu_n,1);
        pred_OSI=NaN(neu_n,1);
        pred_DSI=NaN(neu_n,1);
        % initialize parameter vectors
        alphaopt=NaN(neu_n,1);
        betaopt=NaN(neu_n,1);
        % initialize prediction metrics vectors
        pred_FR=zeros(stim_dur_bins,length(DIR),neu_n,numel(stim_types));
        pred_COUNT=zeros(length(DIR),neu_n,numel(stim_types));
        explvar_D=zeros(neu_n,numel(stim_types));
        explvar_T=zeros(neu_n,numel(stim_types));
        % initialize binning variables
        stimulus_binnumber=floor(stim_dur/frame_dur);
        isi_binnumber=prediction_isi;
        prediction_binnumber=isi_binnumber + stimulus_binnumber + isi_binnumber;
        
        % load RF anlysis results for current layer
        load([resultpath,filesep,'RFs_datastructure_',current_layer_name,'.mat']);
        % fetch RFs to use
        current_cells_idx=1:neu_n; % always all of them
        filters_to_use_raw=Zwsta(:,:,:,current_cells_idx);
        filters_to_use=Zwsta(:,:,:,current_cells_idx);
        % cis_per_frame_to_use=contrast(:,current_cells_idx); NB: add contrast factor analysis before
        
        % loop over selected neurons
        for i=1:neu_n
            
            % get back STA filter
            selectedSTA=filters_to_use(:,:,:,i);
            selectedSTA_raw=filters_to_use_raw(:,:,:,i);
            n=i; % here they coincide by design
            stimtoplot=cell(1,numel(stim_types));
            rastertoplot=cell(1,numel(stim_types));
            pdirtoplot=cell(1,numel(stim_types));
            psftoplot=cell(1,numel(stim_types));
            ptftoplot=cell(1,numel(stim_types));
            
            % loop over stimulus types
            for k=1:numel(stim_types)
                
                tic
                
                % set id of condition to plot ( to default value in this case )
                pdirtoplot{k}=pDIR_O(i);
                psftoplot{k}=SF;
                ptftoplot{k}=TF;
                % set nonlinearity parameters ( to default value in this case )
                alphaopt(i)=1;
                betaopt(i)=1;
                
                % fetch original filter - (preprocessing filters)
                input_interp_factor=interpolation_factor;
                input_STA=selectedSTA((STAborder+1):(STAheight-STAborder),(STAborder+1):(STAwidth-STAborder),:);
                % initialize current interpolated filter
                input_filter = zeros(...
                    size(input_STA,1)*input_interp_factor,...
                    size(input_STA,2)*input_interp_factor,...
                    size(input_STA,3));
                % loop over frames
                for ff=1:size(input_filter,3)
                    % get current original filter frame
                    input_fr = input_STA(:,:,ff);
                    input_fr_interp = interpolate_RF_frame( input_fr, input_interp_factor );
                    % stack into interpolated filter matrix
                    input_filter(:,:,ff) = input_fr_interp;
                end
                % reshaping filter for fast convolution and notmalizing
                STAf=input_filter-mean(mean(mean(input_filter)));
                STAf=STAf./(max(STAf(:))-min(STAf(:)));
                nSTA_frames=4;
                flipped_STAf=flip(STAf(:,:,1:nSTA_frames),3);
                input_F=flipped_STAf(:);
                
                % fetch interp factor - (preprocessing stim)
                input_interp_factor=interpolation_factor;
                % %             % initialize interpolated stims cell
                % %             input_stim = cell(1,size(Stimuli{k},1));
                % %             % loop over directions
                % %             for gg=1:size(Stimuli{k},1)
                % %                 % initialize current interpolated stim
                % %                 tempstim = zeros(...
                % %                     size(input_STA,1)*input_interp_factor,...
                % %                     size(input_STA,2)*input_interp_factor,...
                % %                     2*size(Stimuli{k},2));
                % %                 % loop over frames
                % %                 for hh=1:size(Stimuli{k},2)
                % %                     % get current original filter frame
                % %                     input_fr = squeeze(Stimuli{k}(gg,hh,1:60,1:60));
                % %                     input_fr_interp = interpolate_RF_frame( input_fr, size(input_STA,1)*input_interp_factor/size(input_fr,1) );
                % %                     % stack into interpolated filter matrix
                % %                     tempstim(:,:,hh+size(Stimuli{k},2)) = input_fr_interp;
                % %                 end
                % %                 if pDIR_O(i)==DIR(gg)
                % %                     stimtoplot{k}=tempstim(:,:,(1+size(Stimuli{k},2)):end);
                % %                 end
                % %                 % add blanck padding at the beginning of each stim
                % %                 tempstim(:,:,1:size(Stimuli{k},2))=zeros(size(tempstim(:,:,size(Stimuli{k},2)+1:end)));
                % %             end
                % reshaping stimulus for fast convolution
                num_conv_steps=(2*size(Stimuli{k},2))-size(selectedSTA,3);
                num_stimblock_elements=numel(input_F);
                input_S=cell(1,length(DIR));
                for dir_idx=1:length(DIR)
                    
                    % initialize S matrix
                    input_S{dir_idx}=NaN(num_stimblock_elements,num_conv_steps);
                    
                    % initialize current interpolated stim
                    tempstim = zeros(...
                        size(input_STA,1)*input_interp_factor,...
                        size(input_STA,2)*input_interp_factor,...
                        round(2*size(Stimuli{k},2)));
                    % loop over frames
                    for hh=1:size(Stimuli{k},2)
                        % get current original filter frame
                        input_fr = squeeze(Stimuli{k}(dir_idx,hh,1:round((size(Stimuli{k},3)*0.66)),1:round((size(Stimuli{k},4)*0.66))));
                        % input_fr = squeeze(Stimuli{k}(dir_idx,hh,:,:));
                        input_fr_interp = interpolate_RF_frame( input_fr, size(input_STA,1)*input_interp_factor/size(input_fr,1) );
                        % stack into interpolated filter matrix
                        tempstim(:,:,hh+size(Stimuli{k},2)) = input_fr_interp;
                    end
                    if pDIR_O(i)==DIR(dir_idx)
                        stimtoplot{k}=tempstim(:,:,(1+size(Stimuli{k},2)):end);
                    end
                    % add blanck padding at the beginning of each stim
                    tempstim(:,:,1:size(Stimuli{k},2))=zeros(size(tempstim(:,:,size(Stimuli{k},2)+1:end)));
                    
                    for kk=1:((2*size(Stimuli{k},2))-nSTA_frames)
                        % unroll current stimblock
                        input_S{dir_idx}(:,kk)=single(reshape(tempstim(:,:,(1:nSTA_frames)+(kk-1)),[1,num_stimblock_elements]));
                    end
                    clear tempstim
                end
                clear input_stim
                
                % initialize common neural filteing inputs
                alpha=alphaopt(i);
                beta=betaopt(i);
                sr=30;
                countingwindowlim=[0.9,2];
                pcount=NaN(1,length(DIR));
                prate=NaN(size(input_S{dir_idx},2)+10,length(DIR));
                % loop over directions
                for dir_idx=1:length(DIR)
                    % initialize current neural filteing inputs
                    stimul=input_S{dir_idx};
                    filter=input_F;
                    % perform neural filtering
                    [ prate(:,dir_idx), pcount(dir_idx), pratetime ] = neural_filtering_bis( ...
                        filter,stimul,alpha,beta,sr,countingwindowlim );
                end
                % output progress message
                fprintf(['filter response at ',stim_types{k},...
                    ' ( SF=',SF,' TF=',TF,' ) neuron ',num2str(n),' (',current_layer_name,') computed ...\n']);
                
                % fetch input for goodness of fit
                pred_fr=NaN(size(obs_FR,1),size(obs_FR,2));
                for iiii=1:size(obs_FR,2)
                    vpt=linspace(0,1,size(prate(((end/2)+1):end,:),1) );
                    qpt=linspace(0,1,size(obs_FR,1));
                    vals=prate(((end/2)+1):end,iiii);
                    pred_fr(:,iiii)=interp1(...
                        vpt,...
                        vals,....
                        qpt);
                end
                pred_count=pcount./max(pcount);
                obs_fr=obs_FR(:,:,i,k);
                obs_count=obs_COUNT(:,i,k)';
                % get goodness of fit
                [explvar_T(i,k),rmse_tuning,explvar_D(i,k),rmse_dynamics] = ...
                    get_neuron_responses_prediction_gof(obs_fr,obs_count,pred_fr,pred_count,DIR==pDIR_O(i));
                % store results
                pred_FR(:,:,i,k)=pred_fr;
                pred_COUNT(:,i,k)=pred_count;
                FR_time=pratetime;
                
                % clear heavy vars
                clear input_F input_S
                
                toc
                
                if k==2
                    % get tuning curves
                    tuning_curve_grating_P=pred_COUNT(:,i,1);
                    tuning_curve_plaid_P=pred_COUNT(:,i,2);
                    % perform partial correlation analysis to get predicted index values
                    [ PI_P(i), ~, Zp_P(i), Zc_P(i), ~, ~, ~, ~, ~, ~ ] =...
                        get_pattern_index( tuning_curve_grating_P,tuning_curve_plaid_P );
                    % get predicted pDIR OSI and DSI
                    [ pred_OSI(i),pred_DSI(i),~,~,temp_pDIR_idx  ] = compute_SIs( tuning_curve_grating_P );
                    pDIR_P(i)=DIR(temp_pDIR_idx);
                    % reclassify
                    if Zp_P(i)-max(Zc_P(i),0)>=1.28
                        ctlabel_P(i)=2; % 2=pattern
                    elseif Zc_P(i)-max(Zp_P(i),0)>=1.28
                        ctlabel_P(i)=1; % 1=component
                    else
                        ctlabel_P(i)=0; % 0=unclassified
                    end
                end
                
                tic
                if k==2 % --------------------------------------------------------------
                    for kk=1:2
                        
                        % get pref dir to plot
                        pDIR=pdirtoplot{kk};
                        pSF=psftoplot{kk};
                        pTF=ptftoplot{kk};
                        
                        % initialize figure
                        f1 = figure('units','normalized','outerposition',[0 0 1 1]);
                        % set psth subplot position ----------
                        sb1=subplot(666,666,1);
                        set(sb1,'Position',[.52,0.4,.5,.5]);
                        axis square
                        % get PSTHs to plot
                        psth_predicted=pred_FR(:,DIR==pDIR,i,kk);
                        psth_predicted=psth_predicted./(max(psth_predicted(:)));
                        psth_observed=obs_FR(:,DIR==pDIR,i,kk);
                        psth_observed=psth_observed./(max(psth_observed(:)));
                        % set color and tag to use
                        if ctlabel_O(i)==2
                            coltuse=[255,150,0]./255;
                        elseif ctlabel_O(i)==1
                            coltuse=[50,200,0]./255;
                        elseif ctlabel_O(i)==0
                            coltuse=[150,150,150]./255;
                        end
                        hold on;
                        % draw psth
                        timevec=(1:numel(psth_observed))*(1/sr);
                        plot(gca,timevec,psth_observed,'-','Color',coltuse,'LineWidth',2.5);
                        plot_shaded_auc(gca,timevec,psth_observed',0.15,coltuse)
                        plot(gca,timevec,psth_predicted,':','Color',coltuse*0.5,'LineWidth',2.5);
                        xlim([0,1]);
                        ylim([-0,2]);
                        plot([0,0],[0,5],'--k', 'LineWidth',2)
                        plot([1,1],[0,5],'--k', 'LineWidth',2)
                        tt=text(0.05,3.5,['DIR = ',num2str(pDIR),' d'],'FontSize',12);
                        % ttt=text(0.05,3.25,['spike count = ',num2str(spk)],'FontSize',12);
                        title([stim_types{kk},' psth - neuron ',num2str(n),' - ',layer_names{current_layer_id}]);
                        set(gca,'FontSize',12);
                        tttt = text(0.05,3.00,['ev psth = ',num2str(explvar_D(i,kk),'%.2f')],'FontSize',12);
                        hlabelx=get(gca,'Xlabel');
                        set(hlabelx,'String','time (s)','FontSize',12,'color','k')
                        hlabely=get(gca,'Ylabel');
                        set(hlabely,'String','normalized firing rate','FontSize',12,'color','k')
                        % set psth subplot position ----------
                        ppol=polaraxes('Position',[.02,0.4,.5,.5]);
                        % get tuning curve to plot
                        temp_pred_tc=squeeze(pred_COUNT(:,i,kk))'./max(squeeze(pred_COUNT(:,i,kk)));
                        temp_obs_tc=squeeze(obs_COUNT(:,i,kk))'./max(squeeze(obs_COUNT(:,i,kk)));
                        pred_tc=[temp_pred_tc,temp_pred_tc(1)];
                        obs_tc=[temp_obs_tc,temp_obs_tc(1)];
                        title(ppol,[stim_types{kk},' tuning curve - TF = ',num2str(pTF),' Hz - SF = ',num2str(pSF,'%.2f'),' cpd'],'fontsize',12)
                        set(ppol,'fontsize',12);
                        hold on;
                        % draw plar plots
                        p1=polarplot(ppol,[deg2rad(DIR),2*pi],pred_tc,'-');
                        p2=polarplot(ppol,[deg2rad(DIR),2*pi],obs_tc,'-');
                        set(p1,'color',coltuse*0.5)
                        set(p2,'color',coltuse)
                        set(p1, 'linewidth', 3.5);
                        set(p2, 'linewidth', 3.5);
                        set(ppol,'fontsize',12)
                        tx0=text(ppol,deg2rad(45),1.5,['ev tuning = ',num2str(explvar_T(i,kk),'%.2f')],'fontsize',12);
                        tx1=text(ppol,deg2rad(40),1.5,['betaopt (npTF) = ',num2str(betaopt(i),'%.2f')],'fontsize',12);
                        tx2=text(ppol,deg2rad(30),1.5,['Zp obs = ',num2str(Zp_O(i),'%.01f')],'fontsize',12);
                        tx3=text(ppol,deg2rad(25),1.5,['Zp pred = ',num2str(Zp_P(i),'%.01f')],'fontsize',12);
                        tx4=text(ppol,deg2rad(20),1.5,['Zc obs = ',num2str(Zc_O(i),'%.01f')],'fontsize',12);
                        tx5=text(ppol,deg2rad(15),1.5,['Zc pred = ',num2str(Zc_P(i),'%.01f')],'fontsize',12);
                        tx6=text(ppol,deg2rad(10),1.5,['obs = ',cell_types_names{ctlabel_O(i)+1}],'fontsize',12);
                        tx7=text(ppol,deg2rad(5),1.5,['pred = ',cell_types_names{ctlabel_P(i)+1}],'fontsize',12);
                        % loop over frames to draw filter
                        for jj=1:size(selectedSTA,3)
                            sb3=subplot(666,666,1);
                            fram=selectedSTA_raw((STAborder+1):(STAheight-STAborder),(STAborder+1):(STAwidth-STAborder),jj);
                            fram=imresize(fram,3);
                            l1=imagesc(fram); colormap('gray');
                            caxis([-8,8]); set(gca,'dataAspectRatio',[1 1 1]); axis off
                            set(sb3,'Position',[.02+0.095*(jj-1),0.1,.09,.09]);
                            % text(sb3,(2/3)*size(fram,1),60,['CI=',num2str(cis_per_frame_to_use(jj,i))]);  % NB: added for plotting
                            % text(sb3,(2/3)*size(fram,1),70,['R2=',num2str(r2s_per_frame_to_use(jj,i))]);  % NB: added for plotting
                        end
                        hold on
                        
                        % loop over frames to draw stimulus
                        pstimul=permute(squeeze(Stimuli{kk}(pDIR_O(i)==DIR,1:10,:,:)),[2,3,1]);
                        for jj=1:size(selectedSTA,3)
                            sb4=subplot(666,666,1);
                            fram=stimtoplot{kk}(:,:,jj);
                            imagesc(fram); colormap(gray); set(gca,'dataAspectRatio',[1 1 1]); axis off
                            set(sb4,'Position',[.02+0.095*(jj-1),0.2,.09,.09]);
                        end
                        % save
                        fname=[outputfolder,filesep,'prediction_','n_',num2str(n),'_',current_layer_name,'_',stim_types{kk}];
                        fname=strrep(fname,'.','');
                        
                        % save jpg(gcf,fname, 'epsc')
                        set(gcf, 'PaperPositionMode', 'auto')
                        saveas(gcf,fname, 'jpg')
                        % save eps NB: added for plotting
                        print(gcf,'-depsc','-painters',[fname,'.eps'])
                        
                        toc
                        
                    end  % --------------------------------------------------------------
                    close all
                end
                
            end
            
        end
        
        % save results
        save([outputfolder,filesep,current_layer_name,'_prediction_explained_variance'],...
            'Zp_O',...
            'Zc_O',...
            'pDIR_O',...
            'PI_O',...
            'ctlabel_O',...
            'obs_OSI',...
            'obs_DSI',...
            'obs_FR',...
            'obs_COUNT',...
            'Zp_P',...
            'Zc_P',...
            'pDIR_P',...
            'PI_P',...
            'ctlabel_P',...
            'pred_OSI',...
            'pred_DSI',...
            'obs_FR',...
            'obs_COUNT',...
            'pred_FR',...
            'pred_COUNT',...
            'explvar_D',...
            'explvar_T');
        
    end
    
end

%% collect STA based prediction results from all layers  -------------------------------------------------

% initialize index storage stuctures
PI_Os=cell2struct(cell(size(layer_names)),layer_names);
PI_Ps=cell2struct(cell(size(layer_names)),layer_names);
Zc_Os=cell2struct(cell(size(layer_names)),layer_names);
Zc_Ps=cell2struct(cell(size(layer_names)),layer_names);
Zp_Os=cell2struct(cell(size(layer_names)),layer_names);
Zp_Ps=cell2struct(cell(size(layer_names)),layer_names);
ctlabel_Os=cell2struct(cell(size(layer_names)),layer_names);
ctlabel_Ps=cell2struct(cell(size(layer_names)),layer_names);
explvar_Ds=cell2struct(cell(size(layer_names)),layer_names);
explvar_Ts=cell2struct(cell(size(layer_names)),layer_names);
obs_OSIs=cell2struct(cell(size(layer_names)),layer_names);
obs_DSIs=cell2struct(cell(size(layer_names)),layer_names);
pred_OSIs=cell2struct(cell(size(layer_names)),layer_names);
pred_DSIs=cell2struct(cell(size(layer_names)),layer_names);
Lid_Os=cell2struct(cell(size(layer_names)),layer_names);
obs_COUNTs=cell2struct(cell(size(layer_names)),layer_names);
pred_COUNTs=cell2struct(cell(size(layer_names)),layer_names);

% loop over layers
for current_layer_id=1:numel(layer_names)
    tic
    
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % load results for current layer
    current_results=load([outputfolder,filesep,current_layer_name,'_prediction_explained_variance.mat']);
    % fill-in index storage stuctures
    PI_Os.(current_layer_name)=current_results.PI_O;
    PI_Ps.(current_layer_name)=current_results.PI_P;
    Zc_Os.(current_layer_name)=current_results.Zc_O;
    Zc_Ps.(current_layer_name)=current_results.Zc_P;
    Zp_Os.(current_layer_name)=current_results.Zp_O;
    Zp_Ps.(current_layer_name)=current_results.Zp_P;
    ctlabel_Os.(current_layer_name)=current_results.ctlabel_O;
    ctlabel_Ps.(current_layer_name)=current_results.ctlabel_P;
    explvar_Ds.(current_layer_name)=current_results.explvar_D;
    explvar_Ts.(current_layer_name)=current_results.explvar_T;
    obs_OSIs.(current_layer_name)=current_results.obs_OSI;
    obs_DSIs.(current_layer_name)=current_results.obs_DSI;
    pred_OSIs.(current_layer_name)=current_results.pred_OSI;
    pred_DSIs.(current_layer_name)=current_results.pred_DSI;
    Lid_Os.(current_layer_name)=Lid_observed.(current_layer_name);
    obs_COUNTs.(current_layer_name)=permute(current_results.obs_COUNT,[2,1,3]);
    pred_COUNTs.(current_layer_name)=permute(current_results.pred_COUNT,[2,1,3]);
    
    toc
end

%% plot predicted vs. observed bullet plots (layer mosaic) -------------------------------------------------

% initialize figure
f1 = figure('units','normalized','outerposition',[0 0 1 1]);

% loop over layers
for current_layer_id=1:numel(layer_names)-1
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % get colors
    inputpars.distrcolors{3}=[0,0,0];
    inputpars.distrcolors{1}=[50,200,0]./255;
    inputpars.distrcolors{2}=[255,150,0]./255;
    % set subplot
    subplot(2,3,current_layer_id)
    hold on;
    % loop over neurons
    for current_n=1:numel(Clabel_observed.(current_layer_name))
        % fetch values
        Rlabel=logical(sum(Rlabel_observed.(current_layer_name)(current_n,:)));
        % get observed values
        Clabel_O=ctlabel_Os.(current_layer_name)(current_n);
        Zc_O=Zc_Os.(current_layer_name)(current_n);
        Zp_O=Zp_Os.(current_layer_name)(current_n);
        DSI_O=obs_DSIs.(current_layer_name)(current_n,1);
        % get predicted values
        Clabel_P=ctlabel_Ps.(current_layer_name)(current_n);
        Zc_P=Zc_Ps.(current_layer_name)(current_n);
        Zp_P=Zp_Ps.(current_layer_name)(current_n);
        DSI_P=pred_DSIs.(current_layer_name)(current_n,1);
        
        % if responsive
        if not(sum(Rlabel)==0) && not(Clabel_O==cell_types_codes(3))
            % get cell class idx
            cell_class_idx=find(Clabel_O==cell_types_codes);
            % plot in bullet
            scatter(Zc_O,Zp_O,50,...
                'MarkerFaceColor',inputpars.distrcolors{cell_class_idx},'MarkerEdgeColor',inputpars.distrcolors{cell_class_idx},...
                'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
            scatter(Zc_P,Zp_P,50,...
                'MarkerFaceColor',inputpars.distrcolors{cell_class_idx}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{cell_class_idx}.*0.75,...
                'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
            startvec=[Zc_O,Zp_O];
            endvec=[Zc_P,Zp_P];
            plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{cell_class_idx}*0.75,0.25])
        end
    end
    line([0 10], [1.28 11.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    line([1.28 11.28], [0 10],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    line([1.28 1.28], [-5 0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    line([-5 0], [1.28 1.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    xlim([-5,12])
    ylim([-5,12])
    axis square
    xlabel('Zc'); ylabel('Zp');
    title(['Zp vs. Zc ',current_layer_name,' obs. vs. pred. ( ntot resp =',...
        num2str(sum(logical(sum(Rlabel_observed.(current_layer_name),2)))),' )']);
    set(gca,'fontsize',12)
end
suptitle('DorsalNet - bullet plot - pattern and components across layers - obs. vs pred.')
% save plot
saveas(f1,[outputfolder,filesep,'bullets_across_layers_obs_vs_pred'],'jpg')
print(f1,'-depsc','-painters',[[outputfolder,filesep,'bullets_across_layers_obs_vs_pred'],'.eps'])

%% plot predicted vs. observed bullet plots (all layers together) -------------------------------------------------

% initialize figure
f2 = figure('units','normalized','outerposition',[0 0 1 1]);

% concatenate layer data
Rlabel_Os_concat = concatenate_layers(Rlabel_observed);
ctlabel_Os_concat = concatenate_layers(ctlabel_Os);
ctlabel_Ps_concat = concatenate_layers(ctlabel_Ps);
DSI_Os_concat = concatenate_layers(obs_DSIs);
DSI_Ps_concat = concatenate_layers(pred_DSIs);
Zc_Os_concat = concatenate_layers(Zc_Os);
Zp_Os_concat = concatenate_layers(Zp_Os);
Zc_Ps_concat = concatenate_layers(Zc_Ps);
Zp_Ps_concat = concatenate_layers(Zp_Ps);
PI_Os_concat = concatenate_layers(PI_Os);
PI_Ps_concat = concatenate_layers(PI_Ps);
Lid_Os_concat = concatenate_layers(Lid_Os);
% get target layer boolean
Lidbool_Os_concat = filter_layer(Lid_Os_concat,target_layers_ids);

% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;

% plot bullet plot -----------------------------------------------------
subplot(1,2,1)
hold on;
% loop over neurons
for current_n=1:numel(ctlabel_Os_concat)
    % fetch values
    Rlabel=Rlabel_Os_concat(current_n,:);
    Lid=Lidbool_Os_concat(current_n);
    % get observed values
    Clabel_O=ctlabel_Os_concat(current_n);
    Zc_O=Zc_Os_concat(current_n);
    Zp_O=Zp_Os_concat(current_n);
    DSI_O=DSI_Os_concat(current_n,1);
    % get predicted values
    Clabel_P=ctlabel_Ps_concat(current_n);
    Zc_P=Zc_Ps_concat(current_n);
    Zp_P=Zp_Ps_concat(current_n);
    DSI_P=DSI_Ps_concat(current_n,1);
    % if responsive
    if not(and(sum(Rlabel),Lid)==0) && not(Clabel_O==cell_types_codes(3))
        % get cell class idx
        cell_class_idx=find(Clabel_O==cell_types_codes);
        % plot in bullet
        scatter(Zc_O,Zp_O,50,...
            'MarkerFaceColor',inputpars.distrcolors{cell_class_idx},'MarkerEdgeColor',inputpars.distrcolors{cell_class_idx},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        scatter(Zc_P,Zp_P,50,...
            'MarkerFaceColor',inputpars.distrcolors{cell_class_idx}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{cell_class_idx}.*0.75,...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        startvec=[Zc_O,Zp_O];
        endvec=[Zc_P,Zp_P];
        plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{cell_class_idx}*0.75,0.25])
    end
end
line([0 9], [1.28 10.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
line([1.28 10.28], [0 9],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
line([1.28 1.28], [-5 0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
line([-5 0], [1.28 1.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
xlim([-5,11])
ylim([-5,11])
axis square
xlabel('Zc'); ylabel('Zp');
title(['Zp vs. Zc ','all layers',' obs. vs. pred. ( ntot=',...
    num2str(numel(ctlabel_Os_concat)),' )']);
set(gca,'fontsize',12)
suptitle('DorsalNet - bullet plot - pattern and components (all layers) - obs. vs pred.')
% plot PI violin plot -----------------------------------------------------
subplot(1,2,2)
hold on;
pattern_idx=intersect(find(ctlabel_Os_concat==2),find(Lidbool_Os_concat==1));
component_idx=intersect(find(ctlabel_Os_concat==1),find(Lidbool_Os_concat==1));
distrtoplotlist{1}={...
    [PI_Os_concat(component_idx)],...
    [PI_Os_concat(pattern_idx)],...
    [PI_Ps_concat(component_idx)],...
    [PI_Ps_concat(pattern_idx)]};
ylabellist{1}='pattern index';
yimtouselist{1}=[-14,14];
ks_ban{1}=0.95;
for jj=1
    % decide wheter to use max or unrolled
    distribtouse=distrtoplotlist{jj}; % collapsed_max_fitted_rfs_r2_distribs
    inputpars.inputaxh=gca;
    hold(inputpars.inputaxh,'on')
    % set settings for violin distribution plotting
    inputpars.boxplotwidth=0.4;%0.5;
    inputpars.boxplotlinewidth=2;
    inputpars.densityplotwidth=0.4;%0.5;
    inputpars.yimtouse=yimtouselist{jj};
    % inputpars.yimtouse=[0,8];
    inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
    inputpars.scatteralpha=0.15;
    inputpars.scattersize=40;
    inputpars.distralpha=0.5;
    inputpars.xlabelstring=[];
    inputpars.ylabelstring=ylabellist{jj};
    inputpars.titlestring=[ylabellist{jj},' ( comp # = ',...
        num2str(numel(distribtouse{1}),'%.0f'),...
        ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred','pattern - pred'};
    inputpars.distrcolors{1}=[50,200,0]./255;
    inputpars.distrcolors{2}=[255,150,0]./255;
    inputpars.distrcolors{3}=[50,200,0]./355;
    inputpars.distrcolors{4}=[255,150,0]./355;
    inputaxh = plot_violinplot_PN_new(inputadata,inputpars); %#ok<NASGU>
    pvalw_comp = ranksum(distribtouse{1},distribtouse{3});
    pvalw_patt = ranksum(distribtouse{2},distribtouse{4});
    pvalw_displ = ranksum(distribtouse{1}-distribtouse{3},distribtouse{2}-distribtouse{4});
    text(-0.25,13.5,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
line([-0.5,5.5],[0,0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% save plot
saveas(f2,[outputfolder,filesep,'bullets_all_obs_vs_pred'],'jpg')
print(f2,'-depsc','-painters',[[outputfolder,filesep,'bullets_all_obs_vs_pred'],'.eps'])

%% plot predicted vs. observed Zp and Zc (all layers together) -------------------------------------------------

% initialize figure
f3 = figure('units','normalized','outerposition',[0 0 1 1]);
% set whether to use paired or unpaired testing
pairedtestingbool=1;

% concatenate layer data
Rlabel_Os_concat = concatenate_layers(Rlabel_observed);
ctlabel_Os_concat = concatenate_layers(ctlabel_Os);
ctlabel_Ps_concat = concatenate_layers(ctlabel_Ps);
DSI_Os_concat = concatenate_layers(obs_DSIs);
DSI_Ps_concat = concatenate_layers(pred_DSIs);
Zc_Os_concat = concatenate_layers(Zc_Os);
Zp_Os_concat = concatenate_layers(Zp_Os);
Zc_Ps_concat = concatenate_layers(Zc_Ps);
Zp_Ps_concat = concatenate_layers(Zp_Ps);
PI_Os_concat = concatenate_layers(PI_Os);
PI_Ps_concat = concatenate_layers(PI_Ps);
Lid_Os_concat = concatenate_layers(Lid_Os);
% get target layer boolean
Lidbool_Os_concat = filter_layer(Lid_Os_concat,target_layers_ids);

% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
% plot Zc violin plot -----------------------------------------------------
subplot(1,2,1)
hold on;
pattern_idx=intersect(find(ctlabel_Os_concat==2),find(Lidbool_Os_concat==1));
component_idx=intersect(find(ctlabel_Os_concat==1),find(Lidbool_Os_concat==1));
distrtoplotlist{1}={...
    [Zc_Os_concat(component_idx)],...
    [Zc_Ps_concat(component_idx)],...
    [Zc_Os_concat(pattern_idx)],...
    [Zc_Ps_concat(pattern_idx)]};
ylabellist{1}='Zc';
yimtouselist{1}=[-4,14];
ks_ban{1}=0.75;
for jj=1
    % decide wheter to use max or unrolled
    distribtouse=distrtoplotlist{jj}; % collapsed_max_fitted_rfs_r2_distribs
    inputpars.inputaxh=gca;
    hold(inputpars.inputaxh,'on')
    % set settings for violin distribution plotting
    inputpars.boxplotwidth=0.4;%0.5;
    inputpars.boxplotlinewidth=2;
    inputpars.densityplotwidth=0.4;%0.5;
    inputpars.yimtouse=yimtouselist{jj};
    % inputpars.yimtouse=[0,8];
    inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
    inputpars.scatteralpha=0.15;
    inputpars.scattersize=40;
    inputpars.distralpha=0.5;
    inputpars.xlabelstring=[];
    inputpars.ylabelstring=ylabellist{jj};
    inputpars.titlestring=[ylabellist{jj},' ( comp # = ',...
        num2str(numel(distribtouse{1}),'%.0f'),...
        ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred','pattern - pred'};
    inputpars.distrcolors{1}=[50,200,0]./255;
    inputpars.distrcolors{3}=[255,150,0]./255;
    inputpars.distrcolors{2}=[50,200,0]./355;
    inputpars.distrcolors{4}=[255,150,0]./355;
    [~,scatter_xs,scatter_ys] = plot_violinplot_PN_new(inputadata,inputpars);
    if not(pairedtestingbool)
        pvalw_comp = ranksum(distribtouse{1},distribtouse{2});
        pvalw_patt = ranksum(distribtouse{3},distribtouse{4});
        pvalw_displ = ranksum(distribtouse{1}-distribtouse{2},distribtouse{3}-distribtouse{4});
    else
        pvalw_comp = signrank(distribtouse{1},distribtouse{2});
        pvalw_patt = signrank(distribtouse{3},distribtouse{4});
        pvalw_displ = ranksum(distribtouse{1}-distribtouse{2},distribtouse{3}-distribtouse{4});
    end
    text(-0.25,13.5,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
hold on;
plot([scatter_xs{1},scatter_xs{2}]',[scatter_ys{1},scatter_ys{2}]','linewidth',2,'Color',[inputpars.distrcolors{1}*0.75,0.15])
plot([scatter_xs{3},scatter_xs{4}]',[scatter_ys{3},scatter_ys{4}]','linewidth',2,'Color',[inputpars.distrcolors{3}*0.75,0.15])
line([-0.5,5.5],[0,0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% plot Zp violin plot -----------------------------------------------------
subplot(1,2,2)
hold on;
pattern_idx=intersect(find(ctlabel_Os_concat==2),find(Lidbool_Os_concat==1));
component_idx=intersect(find(ctlabel_Os_concat==1),find(Lidbool_Os_concat==1));
distrtoplotlist{1}={...
    [Zp_Os_concat(component_idx)],...
    [Zp_Ps_concat(component_idx)],...
    [Zp_Os_concat(pattern_idx)],...
    [Zp_Ps_concat(pattern_idx)]};
ylabellist{1}='Zp';
yimtouselist{1}=[-4,14];
ks_ban{1}=0.75;
for jj=1
    % decide wheter to use max or unrolled
    distribtouse=distrtoplotlist{jj}; % collapsed_max_fitted_rfs_r2_distribs
    inputpars.inputaxh=gca;
    hold(inputpars.inputaxh,'on')
    % set settings for violin distribution plotting
    inputpars.boxplotwidth=0.4;%0.5;
    inputpars.boxplotlinewidth=2;
    inputpars.densityplotwidth=0.4;%0.5;
    inputpars.yimtouse=yimtouselist{jj};
    % inputpars.yimtouse=[0,8];
    inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
    inputpars.scatteralpha=0.15;
    inputpars.scattersize=40;
    inputpars.distralpha=0.5;
    inputpars.xlabelstring=[];
    inputpars.ylabelstring=ylabellist{jj};
    inputpars.titlestring=[ylabellist{jj},' ( comp # = ',...
        num2str(numel(distribtouse{1}),'%.0f'),...
        ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred','pattern - pred'};
    inputpars.distrcolors{1}=[50,200,0]./255;
    inputpars.distrcolors{3}=[255,150,0]./255;
    inputpars.distrcolors{2}=[50,200,0]./355;
    inputpars.distrcolors{4}=[255,150,0]./355;
    [~,scatter_xs,scatter_ys] = plot_violinplot_PN_new(inputadata,inputpars);
    if not(pairedtestingbool)
        pvalw_comp = ranksum(distribtouse{1},distribtouse{2});
        pvalw_patt = ranksum(distribtouse{3},distribtouse{4});
        pvalw_displ = ranksum(distribtouse{1}-distribtouse{2},distribtouse{3}-distribtouse{4});
    else
        pvalw_comp = signrank(distribtouse{1},distribtouse{2});
        pvalw_patt = signrank(distribtouse{3},distribtouse{4});
        pvalw_displ = ranksum(distribtouse{1}-distribtouse{2},distribtouse{3}-distribtouse{4});
    end
    text(-0.25,13.5,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
hold on;
plot([scatter_xs{1},scatter_xs{2}]',[scatter_ys{1},scatter_ys{2}]','linewidth',2,'Color',[inputpars.distrcolors{1}*0.75,0.15])
plot([scatter_xs{3},scatter_xs{4}]',[scatter_ys{3},scatter_ys{4}]','linewidth',2,'Color',[inputpars.distrcolors{3}*0.75,0.15])
line([-0.5,5.5],[0,0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% save resuts
saveas(f3,[outputfolder,filesep,'Zp_Zc_scatter_all_obs_vs_pred'],'jpg')
print(f3,'-depsc','-painters',[[outputfolder,filesep,'Zp_Zc_scatter_all_obs_vs_pred'],'.eps'])

%% plot barplot observed vs. predicted category change -------------------------.

% initialize reclassification count (i.e. category change) storage stuctures
reclass_per_class_count=cell2struct(cell(size(layer_names)),layer_names);
reclass_per_class_count_tot=cell(1,3);
% loop over layers
for current_layer_id=1:numel(layer_names)
    if current_layer_id==1
        % initialize total counts
        reclass_per_class_count_tot{1}=zeros(1,3);
        reclass_per_class_count_tot{2}=zeros(1,3);
        reclass_per_class_count_tot{3}=zeros(1,3);
    end
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % get current labels
    labl_P=ctlabel_Ps.(current_layer_name);
    labl_O=ctlabel_Os.(current_layer_name);
    % initialize reclassification count cell array
    reclass_per_class_count.(current_layer_name)=cell(1,3);
    % loop over classes
    for cell_class_idx=1:2
        % set self and null class codes
        if cell_class_idx==1
            current_cell_type_code=1;
            current_cell_type_code_null=2;
        elseif cell_class_idx==2
            current_cell_type_code=2;
            current_cell_type_code_null=1;
        end
        % get original indexes for current class
        original_idx_cc=find(labl_O==cell_class_idx);
        % restrict to original indexes for current class
        labl_P_cc=labl_P(original_idx_cc);
        labl_O_cc=labl_O(original_idx_cc);
        % count reclassifications
        reclass_per_class_count.(current_layer_name){cell_class_idx}(1)=sum(labl_P_cc==current_cell_type_code);
        reclass_per_class_count.(current_layer_name){cell_class_idx}(2)=sum(labl_P_cc==current_cell_type_code_null);
        reclass_per_class_count.(current_layer_name){cell_class_idx}(3)=sum(labl_P_cc==0);
        % accumulate counts
        % if current layer has to be considered
        if logical(sum(filter_layer(Lid_Os.(current_layer_name),target_layers_ids)))
            reclass_per_class_count_tot{cell_class_idx}=reclass_per_class_count_tot{cell_class_idx}+reclass_per_class_count.(current_layer_name){cell_class_idx};
        end
    end
end
% initialize figure
fighand2tris = figure('units','normalized','outerposition',[0 0 1 1]);
% get input for barplot
comp_n_dis=reclass_per_class_count_tot{1};
patt_n_dis=reclass_per_class_count_tot{2};
ytouse=[comp_n_dis./sum(comp_n_dis),patt_n_dis./sum(patt_n_dis)];
xtouse=[1:3,5:7];
%get colors
compcol=[50,200,0]./255;
pattcol=[255,150,0]./255;
% plot bars
hold on;
bar([xtouse([3]),0.1+xtouse([3])],[ytouse([3]),NaN],...
    'facecolor',[0.5,0.5,0.5],...
    'edgecolor',[0.5,0.5,0.5],...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',7) %#ok<*NBRAK>
bar([xtouse([6]),0.1+xtouse([6])],[ytouse([6]),NaN],...
    'facecolor',[0.5,0.5,0.5],...
    'edgecolor',[0.5,0.5,0.5],...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',7)
bar([xtouse([1]),0.1+xtouse([1])],[ytouse([1]),NaN],...
    'facecolor',compcol,...
    'edgecolor',compcol,...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',7)
bar([xtouse([4]),0.1+xtouse([4])],[ytouse([4]),NaN],...
    'facecolor',pattcol,...
    'edgecolor',pattcol,...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',7)
bar([xtouse([2]),0.1+xtouse([2])],[ytouse([2]),NaN],...
    'facecolor',compcol./2,...
    'edgecolor',compcol./2,...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',7)
bar([xtouse([5]),0.1+xtouse([5])],[ytouse([5]),NaN],...
    'facecolor',pattcol./2,...
    'edgecolor',pattcol./2,...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',7)
titlestring=['DorsalNet (all layers) - pred. vs. obs. classification',' - ( #comp = ',...
    num2str(sum(comp_n_dis),'%.0f'),...
    ' - #patt = ',num2str(sum(patt_n_dis),'%.0f'),' )'];
title(titlestring)
ylim([0,0.95])
xlim([0,8])
[chi2stat,p_chitest] = chiSquareTest([comp_n_dis',patt_n_dis']); %#ok<ASGLU>
text(1,0.9,['chi square p = ',num2str(p_chitest)],'fontsize',12);
xticks(xtouse)
xticklabels({'same (cc)','opposite (cp)','unclassified (cu)','same (pp)','opposite (pc)','unclassified (pu)'})
xtickangle(45)
xlabel('')
ylabel('fraction of cells')
set(gca,'fontsize',12)
% save plot
saveas(fighand2tris,[outputfolder,filesep,'pred_obs_classification_barplot_all'],'jpg')
print(fighand2tris,'-depsc','-painters',[[outputfolder,filesep,'pred_obs_classification_barplot_all'],'.eps'])

%% plot cross orientation suppression -------------------------.

% initialize cross orientation suppression index storage structure
CSI_Os=cell2struct(cell(size(layer_names)),layer_names);
% loop over layers
for current_layer_id=1:numel(layer_names)
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % get current labels
    current_TCu=TCu_observed.(current_layer_name);
    pDIR=pDIR_observed.(current_layer_name);
    % initialize temporary cross orientation suppression index storage
    temp_CSI=NaN(1,size(pDIR,1));
    % loop over neurons
    for neu_idx=1:size(pDIR,1)
        % get preferred grating response
        grat_resp=current_TCu(neu_idx,pDIR(neu_idx,1)==DIR,1);
        % get preferred plaid response
        plaid_resp=current_TCu(neu_idx,pDIR(neu_idx,2)==DIR,2);
        % compute cross orientation suppression index
        temp_CSI(neu_idx)=(grat_resp-plaid_resp)./(grat_resp+plaid_resp);
    end
    % store result for current layer
    CSI_Os.(current_layer_name)=temp_CSI';
end
% get concatenated datastructure
CSI_Os_concat = concatenate_layers(CSI_Os);
% organize by cell type
pattern_idx=intersect(find(ctlabel_Os_concat==2),find(Lidbool_Os_concat==1));
component_idx=intersect(find(ctlabel_Os_concat==1),find(Lidbool_Os_concat==1));
unclassified_idx=intersect(find(ctlabel_Os_concat==0),find(Lidbool_Os_concat==1));
CSI_to_use{1}=CSI_Os_concat(component_idx); %#ok<*FNDSB>
CSI_to_use{2}=CSI_Os_concat(pattern_idx);
CSI_to_use{3}=CSI_Os_concat(unclassified_idx);
% plot CSI analysis result
fighand2quater=figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1)
% plot distribution of CSI
nametag='CSI';
% decide wheter to use max or unrolled
distribtouse=CSI_to_use; % collapsed_max_fitted_rfs_r2_distribs
inputpars.inputaxh=gca;
hold(inputpars.inputaxh,'on')
% set settings for violin distribution plotting
inputpars.boxplotwidth=0.4;%0.5;
inputpars.boxplotlinewidth=2;
inputpars.densityplotwidth=0.5;%0.5;
inputpars.yimtouse=[-1.2,1.2];
% inputpars.yimtouse=[0,8];
inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
inputpars.scatteralpha=0.15;
inputpars.scattersize=20;
inputpars.distralpha=0.5;
inputpars.xlabelstring=[];
inputpars.ylabelstring='CSI';
inputpars.titlestring=['cross-suppression index (',nametag,')',' ( comp # = ',...
    num2str(numel(distribtouse{1}),'%.0f'),...
    ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),...
    ' - uncl #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
inputpars.boolscatteron=1;
inputpars.ks_bandwidth=0.1;
inputpars.xlimtouse=[-0.5,4.5]; %[-1,5];
% plot violins
inputadata.inputdistrs=distribtouse;
inputpars.n_distribs=numel(inputadata.inputdistrs);
inputpars.dirstrcenters=(1:inputpars.n_distribs);
inputpars.xtickslabelvector={'component','pattern','unclassified'};
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=[255/2,255/2,255/2]./255;
inputaxh = plot_violinplot_PN_new(inputadata,inputpars); %#ok<NASGU>
% [~,pval_csi_pc] = ttest2(distribtouse{1}',distribtouse{2}');
% [~,pval_csi_c] = ttest(distribtouse{1});
% [~,pval_csi_p] = ttest(distribtouse{2});
% [~,pval_csi_unc] = ttest(distribtouse{3});
[pval_csi_pc,~] = ranksum(distribtouse{1}',distribtouse{2}');
[pval_csi_c,~] = signrank(distribtouse{1});
[pval_csi_p,~] = signrank(distribtouse{2});
[pval_csi_unc,~] = signrank(distribtouse{3});
usedxlim=get(gca,'xlim');
hold on;
plot(gca,usedxlim,[0,0],'--','linewidth',2,'color',[0.5,0.5,0.5])
text(gca,-0.25,-0.9,['comp vs. patt median csi diff p = ',num2str(pval_csi_pc)],'fontsize',12)
text(gca,-0.25,-0.95,['comp median csi p = ',num2str(pval_csi_c)],'fontsize',12)
text(gca,-0.25,-0.8,['patt median csi p = ',num2str(pval_csi_p)],'fontsize',12)
text(gca,-0.25,-0.85,['uncl median csi p = ',num2str(pval_csi_unc)],'fontsize',12)
xtickangle(45)
set(gca,'fontsize',12)
axis square
subplot(1,2,2)
distrstoplot=cell(1,numel(layer_names));
% loop over layers
for current_layer_id=1:numel(layer_names)
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    distrstoplot{current_layer_id}=CSI_Os.(current_layer_name);
end
% plot distribution of CSI
nametag='CSI';
% decide wheter to use max or unrolled
distribtouse=distrstoplot;
inputpars.inputaxh=gca;
hold(inputpars.inputaxh,'on')
% set settings for violin distribution plotting
inputpars.boxplotwidth=0.4;%0.5;
inputpars.boxplotlinewidth=2;
inputpars.densityplotwidth=0.5;%0.5;
inputpars.yimtouse=[-1.2,1.2];
% inputpars.yimtouse=[0,8];
inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
inputpars.scatteralpha=0.15;
inputpars.scattersize=20;
inputpars.distralpha=0.5;
inputpars.xlabelstring=[];
inputpars.ylabelstring='CSI';
inputpars.titlestring=['cross-suppression index (',nametag,')',' per layer'];
inputpars.boolscatteron=1;
inputpars.ks_bandwidth=0.2;
inputpars.xlimtouse=[-0.5,8.5]; %[-1,5];
% plot violins
inputadata.inputdistrs=distribtouse;
inputpars.n_distribs=numel(inputadata.inputdistrs);
inputpars.dirstrcenters=(1:inputpars.n_distribs);
inputpars.xtickslabelvector=layer_names;
inputpars.distrcolors{1}=[255,255,255].*(0.75)./255;
inputpars.distrcolors{2}=[255,255,255].*(0.65)./255;
inputpars.distrcolors{3}=[255,255,255].*(0.55)./255;
inputpars.distrcolors{4}=[255,255,255].*(0.45)./255;
inputpars.distrcolors{5}=[255,255,255].*(0.35)./255;
inputpars.distrcolors{6}=[255,255,255].*(0.25)./255;
inputpars.distrcolors{7}=inputpars.distrcolors{1};
inputaxh = plot_violinplot_PN_new(inputadata,inputpars);
hold on;
usedxlim=get(gca,'xlim');
plot(gca,usedxlim,[0,0],'--','linewidth',2,'color',[0.5,0.5,0.5])
xtickangle(45)
set(gca,'fontsize',12)
axis square
% save plot
saveas(fighand2quater,[outputfolder,filesep,nametag,'_scatter'],'jpg')
print(fighand2quater,'-depsc','-painters',[outputfolder,filesep,nametag,'_scatter','.eps'])
% close all

%% CSI per layer ad class

fighand2quater=figure('units','normalized','outerposition',[0 0 1 1]);
cell_types_codes=[1,2,0];
cell_types_colors{1}=[50,200,0]./255;
cell_types_colors{2}=[255,150,0]./255;
cell_types_colors{3}=[255/2,255/2,255/2]./255;
for cell_type_idx=1:numel(cell_types_codes)
    cell_type_colde=cell_types_codes(cell_type_idx);
    subplot(1,3,cell_type_idx)
    distrstoplot=cell(1,numel(layer_names));
    % loop over layers
    for current_layer_id=1:numel(layer_names)
        % get current layer name
        current_layer_name=layer_names{current_layer_id};
        selected_idx=find(ctlabel_Os.(current_layer_name)==cell_type_colde);
        if not(isempty(selected_idx))
            distrstoplot{current_layer_id}=CSI_Os.(current_layer_name)(selected_idx);
        else
            distrstoplot{current_layer_id}=NaN;
        end
    end
    % plot distribution of CSI
    nametag='CSI';
    % decide wheter to use max or unrolled
    distribtouse=distrstoplot;
    inputpars.inputaxh=gca;
    hold(inputpars.inputaxh,'on')
    % set settings for violin distribution plotting
    inputpars.boxplotwidth=0.4;%0.5;
    inputpars.boxplotlinewidth=2;
    inputpars.densityplotwidth=0.5;%0.5;
    inputpars.yimtouse=[-1.2,1.2];
    % inputpars.yimtouse=[0,8];
    inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
    inputpars.scatteralpha=0.15;
    inputpars.scattersize=20;
    inputpars.distralpha=0.5;
    inputpars.xlabelstring=[];
    inputpars.ylabelstring='CSI';
    inputpars.titlestring=[cell_types_names{cell_type_idx},' cross-suppression index (',nametag,')',' per layer'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=0.2;
    inputpars.xlimtouse=[-0.5,8.5]; %[-1,5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector=layer_names;
    inputpars.distrcolors{1}=cell_types_colors{cell_type_idx}.*(0.95);
    inputpars.distrcolors{2}=cell_types_colors{cell_type_idx}.*(0.85);
    inputpars.distrcolors{3}=cell_types_colors{cell_type_idx}.*(0.75);
    inputpars.distrcolors{4}=cell_types_colors{cell_type_idx}.*(0.65);
    inputpars.distrcolors{5}=cell_types_colors{cell_type_idx}.*(0.55);
    inputpars.distrcolors{6}=cell_types_colors{cell_type_idx}.*(0.45);
    inputpars.distrcolors{7}=inputpars.distrcolors{1};
    inputaxh = plot_violinplot_PN_new(inputadata,inputpars);
    hold on;
    usedxlim=get(gca,'xlim');
    plot(gca,usedxlim,[0,0],'--','linewidth',2,'color',[0.5,0.5,0.5])
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
% save plot
saveas(fighand2quater,[outputfolder,filesep,nametag,'_scatter_per_layer_and_class'],'jpg')
print(fighand2quater,'-depsc','-painters',[outputfolder,filesep,nametag,'_scatter_per_layer_and_class','.eps'])
close all

%% collect RF shape features all layers  -------------------------------------------------

% initialize shape features storage stuctures
bestlobenumbers=cell2struct(cell(size(layer_names)),layer_names);
contrasts=cell2struct(cell(size(layer_names)),layer_names);
gaborr2s=cell2struct(cell(size(layer_names)),layer_names);

% loop over layers
for current_layer_id=1:numel(layer_names)
    
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % load results for current layer
    current_results=load([resultpath,filesep,'RFs_datastructure_',current_layer_name,'.mat']);
    
    % get best frame indexes
    [~,bestrfidx]=max(current_results.contrast_factors,[],1);
    
    % initialize best frame lobecount vector
    bestlobenumbers.(current_layer_name)=NaN(size(current_results.lobe_numbers,2),1);
    % loop over neurons
    for neu_idx=1:size(current_results.lobe_numbers,2)
        % fill-in best frame lobecount
        bestlobenumbers.(current_layer_name)(neu_idx)=current_results.lobe_numbers(bestrfidx(neu_idx),neu_idx);
    end
    % fill-in other shape features storage stuctures
    contrasts.(current_layer_name)=current_results.contrast_factors(1:4,:)';
    gaborr2s.(current_layer_name)=current_results.Zwsta_fit_r2(1:4,:)';
    
end

% get concatenated datastructure
bestlobenumbers_concat = concatenate_layers(bestlobenumbers);
contrasts_concat = concatenate_layers(contrasts);
gaborr2s_concat = concatenate_layers(gaborr2s);

% organize by cell type
pattern_idx=intersect(find(ctlabel_Os_concat==2),find(Lidbool_Os_concat==1));
component_idx=intersect(find(ctlabel_Os_concat==1),find(Lidbool_Os_concat==1));
unclassified_idx=intersect(find(ctlabel_Os_concat==0),find(Lidbool_Os_concat==1));
bestlobenumbers_to_use{1}=bestlobenumbers_concat(component_idx);
bestlobenumbers_to_use{2}=bestlobenumbers_concat(pattern_idx);
bestlobenumbers_to_use{3}=bestlobenumbers_concat(unclassified_idx);
contrasts_concat_to_use{1}=reshape(contrasts_concat(component_idx,:), [], 1);
contrasts_concat_to_use{2}=reshape(contrasts_concat(pattern_idx,:), [], 1);
contrasts_concat_to_use{3}=reshape(contrasts_concat(unclassified_idx,:), [], 1);
gaborr2s_to_use{1}=reshape(gaborr2s_concat(component_idx,:), [], 1);
gaborr2s_to_use{2}=reshape(gaborr2s_concat(pattern_idx,:), [], 1);
gaborr2s_to_use{3}=reshape(gaborr2s_concat(unclassified_idx,:), [], 1);
% plot distributions of shape features
distrtoplotlist_orig{1}=contrasts_concat_to_use;
distrtoplotlist_orig{2}=gaborr2s_to_use;
distrtoplotlist_orig{3}=bestlobenumbers_to_use;
distrtoplotlist{1}=distrtoplotlist_orig{1};
distrtoplotlist{2}=distrtoplotlist_orig{2};
distrtoplotlist{3}=distrtoplotlist_orig{3};
ylabellist{1}='contrast factor';
ylabellist{2}='Gabor r2';
yimtouselist{1}=[0,40];
yimtouselist{2}=[0.025,1.1];
ks_ban{1}=3;
ks_ban{2}=0.03;
% initialize figure
fighand2=figure('units','normalized','outerposition',[0 0 1 1]);
for jj=1:2
    % decide wheter to use max or unrolled
    distribtouse=distrtoplotlist{jj}; % collapsed_max_fitted_rfs_r2_distribs
    subplot(1,3,jj)
    inputpars.inputaxh=gca;
    hold(inputpars.inputaxh,'on')
    % set settings for violin distribution plotting
    inputpars.boxplotwidth=0.4;%0.5;
    inputpars.boxplotlinewidth=2;
    inputpars.densityplotwidth=0.4;%0.5;
    inputpars.yimtouse=yimtouselist{jj};
    % inputpars.yimtouse=[0,8];
    inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
    inputpars.scatteralpha=0.15;
    inputpars.scattersize=20;
    inputpars.distralpha=0.5;
    inputpars.xlabelstring=[];
    inputpars.ylabelstring=ylabellist{jj};
    inputpars.titlestring=[ylabellist{jj},' ( #comp fr = ',...
        num2str(numel(distribtouse{1}),'%.0f'),...
        ' - #patt fr = ',num2str(numel(distribtouse{2}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj}; % inputpars.ks_bandwidth=0.25;
    inputpars.xlimtouse=[-0.5,4.5]; %[-1,5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component','pattern','unclassified'};
    inputpars.distrcolors{1}=[50,200,0]./255;
    inputpars.distrcolors{2}=[255,150,0]./255;
    inputpars.distrcolors{3}=[0,0,0]./(3*255);
    inputaxh = plot_violinplot_PN_new(inputadata,inputpars);
    pvalw = ranksum(distribtouse{1},distribtouse{2});
    [junk,pvalt] = ttest2(distribtouse{1},distribtouse{2});
    text(-0.2,0.95*max(yimtouselist{jj}),['median diff p = ',num2str(pvalw)],'fontsize',12);
    text(-0.2,0.92*max(yimtouselist{jj}),['mean diff p = ',num2str(pvalt)],'fontsize',12);
    set(gca,'fontsize',12)
    axis square
end
subplot(1,3,3)
comp_lobe_n_dis=distrtoplotlist{3}{1};
patt_lobe_n_dis=distrtoplotlist{3}{2};
temp=[comp_lobe_n_dis;patt_lobe_n_dis];
xtouse=[min(temp):1:max(temp)];
[N_comp_lobe_n,~] = hist(comp_lobe_n_dis,xtouse); %#ok<HIST>
p_comp_lobe_n=N_comp_lobe_n./sum(N_comp_lobe_n);
[N_patt_lobe_n,X] = hist(patt_lobe_n_dis,xtouse); %#ok<HIST>
p_patt_lobe_n=N_patt_lobe_n./sum(N_patt_lobe_n);
hold on;
bar(10*X,p_comp_lobe_n,...
    'facecolor',inputpars.distrcolors{1},...
    'edgecolor',[0,0,0],...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',0.3)
bar(10*X+3,p_patt_lobe_n,...
    'facecolor',inputpars.distrcolors{2},...
    'edgecolor',[0,0,0],...
    'linewidth',1.5,...
    'facealpha',0.5,...
    'barwidth',0.3)
titlestring=['lobe count',' ( #comp fr = ',...
    num2str(numel(distrtoplotlist{3}{1}),'%.0f'),...
    ' - #patt fr = ',num2str(numel(distrtoplotlist{3}{2}),'%.0f'),' )'];
title(titlestring)
ylim([0,1])
[chi2stat,p_chitest] = chiSquareTest([N_comp_lobe_n;N_patt_lobe_n]);
text(3,0.92,['chi square p = ',num2str(p_chitest)],'fontsize',12);
xticks(10*X+1.5)
xticklabels(X)
xlabel('# of lobes')
ylabel('fraction of cells')
set(gca,'fontsize',12)
axis square
suptitle(['RF shape analysis - all areas'])
saveas(fighand2,[outputfolder,filesep,'RF_shape_analysis_all_areas'],'jpg')
print(fighand2,'-depsc','-painters',[[outputfolder,filesep,'RF_shape_analysis_all_areas'],'.eps'])
close all

%% perform rdm analysis and store representations -------------------------.
% NB:revision additions from here on (26/06/2023)

% get concatenated observed counts datastructure
obs_COUNTs_concat = concatenate_layers(obs_COUNTs);
% get concatenated predicted counts datastructure
pred_COUNTs_concat = concatenate_layers(pred_COUNTs);
% get number of stim conditions
ndir=size(obs_COUNTs_concat,2);
ntype=size(obs_COUNTs_concat,3);

% organize by cell class
pattern_idx=intersect(find(ctlabel_Os_concat==2),find(Lidbool_Os_concat==1));
component_idx=intersect(find(ctlabel_Os_concat==1),find(Lidbool_Os_concat==1));
unclassified_idx=intersect(find(ctlabel_Os_concat==0),find(Lidbool_Os_concat==1));
ncomp=numel(component_idx);
npatt=numel(pattern_idx);
nuncl=numel(unclassified_idx);
n_per_class=[ncomp,npatt,nuncl];
obs_COUNT_per_class{1}=reshape(obs_COUNTs_concat(component_idx,:,:),[ncomp,ndir*ntype]);
obs_COUNT_per_class{2}=reshape(obs_COUNTs_concat(pattern_idx,:,:),[npatt,ndir*ntype]);
obs_COUNT_per_class{3}=reshape(obs_COUNTs_concat(unclassified_idx,:,:),[nuncl,ndir*ntype]);
pred_COUNT_per_class{1}=reshape(pred_COUNTs_concat(component_idx,:,:),[ncomp,ndir*ntype]);
pred_COUNT_per_class{2}=reshape(pred_COUNTs_concat(pattern_idx,:,:),[npatt,ndir*ntype]);
pred_COUNT_per_class{3}=reshape(pred_COUNTs_concat(unclassified_idx,:,:),[nuncl,ndir*ntype]);

% set smoothpar for represenattion preprocessing smoothpar
smoothpar=0.5;
smoothbool=1;

% get matrix of median difference surprises
obs_RDM_per_class=cell(size(1,numel(n_per_class)));
pred_RDM_per_class=cell(size(1,numel(n_per_class)));
for class_idx=1:numel(n_per_class)
    % initialize rdm for current class
    obs_RDM_per_class{class_idx}=NaN(ndir*ntype,ndir*ntype);
    pred_RDM_per_class{class_idx}=NaN(ndir*ntype,ndir*ntype);
    % ge preprocessed representations
    if smoothbool
        obs_rep_mat=max_normalize_halves(gaussianSmooth1D(obs_COUNT_per_class{class_idx}, smoothpar, 1));
        pred_rep_mat=max_normalize_halves(gaussianSmooth1D(pred_COUNT_per_class{class_idx}, smoothpar, 1));
    else
        obs_rep_mat=obs_COUNT_per_class{class_idx}; 
        pred_rep_mat=pred_COUNT_per_class{class_idx};
    end
    for cond1_idx=1:ndir*ntype
        for cond2_idx=1:ndir*ntype
            % compute current rdm element for current class - observed
            curr_obs_rep1=obs_rep_mat(:,cond1_idx);
            curr_obs_rep2=obs_rep_mat(:,cond2_idx);
            curr_corrval=corr(curr_obs_rep1,curr_obs_rep2);
            obs_RDM_per_class{class_idx}(cond1_idx,cond2_idx)=1-curr_corrval;
            % compute current rdm element for current class - predicted
            curr_pred_rep1=pred_rep_mat(:,cond1_idx);
            curr_pred_rep2=pred_rep_mat(:,cond2_idx);
            curr_corrval=corr(curr_pred_rep1,curr_pred_rep2);
            pred_RDM_per_class{class_idx}(cond1_idx,cond2_idx)=1-curr_corrval;
        end
    end
end

% prepare lablels for plotting rdms
labels = cell(ndir,ntype);
ordertags = NaN(ndir,ntype);
for i = 1:12
    for j = 1:2
        number = num2str((i-1)*30);
        if j == 1
            letter = 'g';
        else
            letter = 'p';
        end
        labels{i,j} = [number, letter];
        ordertags(i,j) = i+0.5*(j-1);
    end
end
labels=labels(:);
% get ideal (i.e. pattern) reordering permutation and block labels
ordertags=ordertags(:);
[~,ideal_sorting_perm] = sort(ordertags);
classlabels={'component','pattern','unclassified'};
blocklabels=floor(ordertags);

% inpect pattern and component representations - observed  ----------------
fighand3=figure('units','normalized','outerposition',[0 0 1 1]);
vartypes={'obs comp','obs patt'};
for class_idx=1:(numel(n_per_class)-1)
    % get current imputs
    mat_to_use=obs_RDM_per_class{class_idx};
    colortouse=inputpars.distrcolors{class_idx};
    vartype=vartypes{class_idx};
    % compute block modularity - pattern
    current_blk_mod = compute_block_modularity_PN(mat_to_use, blocklabels);
    % plot dendrogram
    subplot(2,2,2+(class_idx-1)*2)
    var_Z = linkage(mat_to_use,'average','euclidean');
    % perform hierarchical clustering
    [h,~,var_outperm] = dendrogram(var_Z);
    % compute block modularity - best
    dendrogram_height_cutoff = 0.9;
    var_T = cluster(var_Z, 'cutoff', dendrogram_height_cutoff, 'criterion', 'distance');
    current_blk_mod_best = compute_block_modularity_PN(mat_to_use, var_T);
    hold on
    plot([0,ndir*ntype],[dendrogram_height_cutoff,dendrogram_height_cutoff],'--','Linewidth',2,'Color',colortouse)
    xticklabels(gca,labels(var_outperm))
    set( h, 'Color', 'k' );
    set( h, 'LineWidth', 3 );
    set(gca,'fontsize',12)
    xlim(gca,[0,ndir*ntype]);
    if smoothbool
        ylim(gca,[0,3.5]);
    else
        ylim(gca,[0,2.75]) %#ok<*UNRCH>
    end
    xtickangle(gca,30)
    title([vartype,' - HC dendrogram - b mod = ',num2str(round(current_blk_mod_best,3))])
    % plot rdm
    sorting_perm_to_use=ideal_sorting_perm;
    subplot(2,2,1+(class_idx-1)*2)
    mat_to_use_reord=mat_to_use(sorting_perm_to_use,:);
    mat_to_use_reord=mat_to_use_reord(:,sorting_perm_to_use);
    overall_labels_reord=labels(sorting_perm_to_use);
    hold on;
    im1=imagesc(flipud(mat_to_use_reord)); colormap('gray');
    axx=get(im1).Parent;
    if smoothbool
        caxis(gca,[0,1.5]);
    else
        caxis(gca,[0,1])
    end
    set(axx,'YTick',1:numel(overall_labels_reord));
    set(axx,'XTick',1:numel(overall_labels_reord));
    xlim(axx,[1-0.5,numel(overall_labels_reord)+0.5]);
    ylim(axx,[1-0.5,numel(overall_labels_reord)+0.5]);
    set(axx,'YTickLabel',flipud(overall_labels_reord));
    set(axx,'XTickLabel',overall_labels_reord);
    cb1=colorbar;
    ylabel(cb1,'dissimilarity (1-corr)')
    set(cb1,'fontsize',14)
    xtickangle(90)
    set(axx,'fontsize',12)
    axis square
    title([vartype,' - p-reordered RDM - p mod = ',num2str(round(current_blk_mod,3))])
end
suptitle('DorsalNet - patt and comp observed representations - all layers')
saveas(fighand3,[outputfolder,filesep,'DorsalNet_obs_rdm'],'jpg')
print(fighand3,'-depsc','-painters',[[outputfolder,filesep,'DorsalNet_obs_rdm'],'.eps'])
close all

% inpect pattern and component representations - predicted ----------------
fighand4=figure('units','normalized','outerposition',[0 0 1 1]);
vartypes={'pred comp','pred patt'};
for class_idx=1:(numel(n_per_class)-1)
    % get current imputs
    mat_to_use=pred_RDM_per_class{class_idx};
    colortouse=inputpars.distrcolors{class_idx};
    vartype=vartypes{class_idx};
    % compute block modularity
    current_blk_mod = compute_block_modularity_PN(mat_to_use, blocklabels);
    % plot dendrogram
    subplot(2,2,2+(class_idx-1)*2)
    var_Z = linkage(mat_to_use,'average','euclidean');
    % perform hierarchical clustering
    [h,~,var_outperm] = dendrogram(var_Z);
    % compute block modularity - best
    dendrogram_height_cutoff = 1;
    var_T = cluster(var_Z, 'cutoff', dendrogram_height_cutoff, 'criterion', 'distance');
    current_blk_mod_best = compute_block_modularity_PN(mat_to_use, var_T);
    hold on
    plot([0,ndir*ntype],[dendrogram_height_cutoff,dendrogram_height_cutoff],'--','Linewidth',2,'Color',colortouse)
    xticklabels(gca,labels(var_outperm))
    set( h, 'Color', 'k' );
    set( h, 'LineWidth', 3 );
    set(gca,'fontsize',12)
    xlim(gca,[0,ndir*ntype]);
    if smoothbool
        ylim(gca,[0,3.5]);
    else
        ylim(gca,[0,2.75]) %#ok<*UNRCH>
    end
    xtickangle(gca,30)
    title([vartype,' - HC dendrogram - b mod = ',num2str(round(current_blk_mod_best,3))])
    % plot rdm
    sorting_perm_to_use=ideal_sorting_perm;
    subplot(2,2,1+(class_idx-1)*2)
    mat_to_use_reord=mat_to_use(sorting_perm_to_use,:);
    mat_to_use_reord=mat_to_use_reord(:,sorting_perm_to_use);
    overall_labels_reord=labels(sorting_perm_to_use);
    hold on;
    im1=imagesc(flipud(mat_to_use_reord)); colormap('gray');
    axx=get(im1).Parent;
    if smoothbool
        caxis(gca,[0,1.5]);
    else
        caxis(gca,[0,1])
    end
    set(axx,'YTick',1:numel(overall_labels_reord));
    set(axx,'XTick',1:numel(overall_labels_reord));
    xlim(axx,[1-0.5,numel(overall_labels_reord)+0.5]);
    ylim(axx,[1-0.5,numel(overall_labels_reord)+0.5]);
    set(axx,'YTickLabel',flipud(overall_labels_reord));
    set(axx,'XTickLabel',overall_labels_reord);
    cb1=colorbar;
    ylabel(cb1,'dissimilarity (1-corr)')
    set(cb1,'fontsize',14)
    xtickangle(90)
    set(axx,'fontsize',12)
    axis square
    title([vartype,' - p-reordered RDM - p mod = ',num2str(round(current_blk_mod,3))])
end
suptitle('DorsalNet patt and comp LN-predicted representations - all layers')
saveas(fighand4,[outputfolder,filesep,'DorsalNet_pred_rdm'],'jpg')
print(fighand4,'-depsc','-painters',[[outputfolder,filesep,'DorsalNet_pred_rdm'],'.eps'])
close all

% save results of rdm analysis
save([resultpath,filesep,'RDM_datastructure_DorsalNet.mat'],...
    'obs_COUNTs_concat',...
    'pred_COUNTs_concat',...
    'pattern_idx',...
    'component_idx',...
    'unclassified_idx',...
    'obs_COUNT_per_class',...
    'pred_COUNT_per_class',...
    'obs_RDM_per_class',...
    'pred_RDM_per_class',...
    'labels',...
    'classlabels',...
    'blocklabels',...
    'ideal_sorting_perm');

%% analyze plaid & grating modulation indeces and produce raster examples 

% get concatenated cell identity labels and pasth data
mod_ctlabel_concat = concatenate_layers(ctlabel_Os);
mod_psth_concat = concatenate_layers(PSTH_observed);
mod_count_concat = concatenate_layers(obs_COUNTs);
% create a cell array of field names
initialfieldNames = layer_names;
initialValues = cellfun(@(x) [], initialfieldNames, 'UniformOutput', false);
layerlabel_Os = cell2struct(initialValues, initialfieldNames, 1);
withinlayeridx_Os = cell2struct(initialValues, initialfieldNames, 1);
% loop over layers
for current_layer_id=1:numel(layer_names)
    layerlabel_Os.(layer_names{current_layer_id})=current_layer_id.*ones(size(ctlabel_Os.(layer_names{current_layer_id})));
    withinlayeridx_Os.(layer_names{current_layer_id})=(current_layer_id.*(1:length(ctlabel_Os.(layer_names{current_layer_id}))))';
end
% get layer and within layer labels
mod_layerlabel_concat=concatenate_layers(layerlabel_Os);
mod_withinlayeridx_concat=concatenate_layers(withinlayeridx_Os);

% initialize per cell type structures
mod_psth_per_type=cell(1,numel(cell_types_codes));
mod_count_per_type=cell(1,numel(cell_types_codes));
mod_nidx_per_type=cell(1,numel(cell_types_codes));
mod_Zp_per_type=cell(1,numel(cell_types_codes));
mod_Zc_per_type=cell(1,numel(cell_types_codes));
mod_layerlabel_per_type=cell(1,numel(cell_types_codes));
mod_withinlayeridx_per_type=cell(1,numel(cell_types_codes));
% seregate per cell class type
for cell_types_idx=1:numel(cell_types_codes)
    % get indexes of currently selected units
    current_cell_types_code=cell_types_codes(cell_types_idx);
    current_units_idx=find(mod_ctlabel_concat==current_cell_types_code);
    % select psths
    mod_psth_per_type{cell_types_idx}=mod_psth_concat(current_units_idx,:,:);
    mod_count_per_type{cell_types_idx}=mod_count_concat(current_units_idx,:,:);
    mod_nidx_per_type{cell_types_idx}=current_units_idx;
    mod_Zp_per_type{cell_types_idx}=Zp_Os_concat(current_units_idx);
    mod_Zc_per_type{cell_types_idx}=Zc_Os_concat(current_units_idx);
    mod_layerlabel_per_type{cell_types_idx}=mod_layerlabel_concat(current_units_idx);
    mod_withinlayeridx_per_type{cell_types_idx}=mod_withinlayeridx_concat(current_units_idx);
end

% set sampling rate
sr=SR;
cell_types_names=cell_types_names([2,3,1]);
% set whether to plot modulation diagnostics
bool_plot_mod_diagnostics=0;
bool_rerun_mod_analysis=0;
% in needed run MI analysis
if not(and(exist([resultpath,filesep,'MI_results_','DorsalNet','.mat']),not(bool_rerun_mod_analysis))) %#ok<EXIST>
    % initialize mod structures
    mod_MIf1f0=cell(1,numel(cell_types_codes));
    mod_MIf1z=cell(1,numel(cell_types_codes));
    mod_spectrum=cell(1,numel(cell_types_codes));
    mod_FR=cell(1,numel(cell_types_codes));
    mod_pDIR=cell(1,numel(cell_types_codes));
    % loop over cell classes
    for cell_types_idx=1:numel(cell_types_codes)
        % get current nuber of unis
        current_ns=size(mod_psth_per_type{cell_types_idx},1);
        % initialize output matrices
        mod_MIf1f0{cell_types_idx}=cell(1,numel(current_ns));
        mod_MIf1z{cell_types_idx}=cell(1,numel(current_ns));
        mod_spectrum{cell_types_idx}=cell(1,numel(current_ns));
        mod_FR{cell_types_idx}=cell(1,numel(current_ns));
        mod_pDIR{cell_types_idx}=cell(1,numel(current_ns));
        % loop over units
        for i=1:current_ns
            % initialize storage structures
            mod_MIf1f0{cell_types_idx}{i}=NaN(length(DIR),numel(stimulus_types));
            mod_MIf1z{cell_types_idx}{i}=NaN(length(DIR),numel(stimulus_types));
            mod_spectrum{cell_types_idx}{i}=cell(length(DIR),numel(stimulus_types));
            mod_FR{cell_types_idx}{i}=NaN(32,length(DIR),numel(stimulus_types));
            mod_pDIR{cell_types_idx}{i}=NaN(1,numel(stimulus_types));
            % get grating pref direction (NB: everything will be computed in this direction)
            temp=squeeze(mod_count_per_type{cell_types_idx}(i,:,1));
            [tempmaxval,tempmaxid]=max(temp);
            mod_pDIR{cell_types_idx}{i}(1)=DIR(tempmaxid);
            tic
            % loop over stim types
            for k=1:numel(stimulus_types)
                % compute phase modulation
                psth=squeeze(mod_psth_per_type{cell_types_idx}(i,:,k)); % (NB: this is the psth at grating pref dir)
                if not(sum(isnan(psth))==length(psth))
                psth_time=(0:length(psth)).*(1/sr);
                tf_target=2;
                boolplot=0;
                [ current_MIf1z, current_MIf1f0 , current_MIplothandle, current_MIplotdata ] = ...
                    get_modulation_index( psth, max(psth_time), tf_target, boolplot );
                % store results
                mod_FR{cell_types_idx}{i}(:,mod_pDIR{cell_types_idx}{i}(1)==DIR,k)=psth;
                mod_MIf1f0{cell_types_idx}{i}(mod_pDIR{cell_types_idx}{i}(1)==DIR,k)=current_MIf1f0;
                mod_MIf1z{cell_types_idx}{i}(mod_pDIR{cell_types_idx}{i}(1)==DIR,k)=current_MIf1z;
                mod_spectrum{cell_types_idx}{i}{mod_pDIR{cell_types_idx}{i}(1)==DIR,k}=current_MIplotdata;
                end
            end
            toc
            if bool_plot_mod_diagnostics
                % initialize figure
                ff = figure('units','normalized','outerposition',[0 0 1 1]);
                % plot psth and raster ----------------------------
                axis square
                % get data to plot for current neuron
                curr_pDIR=mod_pDIR{cell_types_idx}{i}(1);
                curr_COUNT=squeeze(mod_count_per_type{cell_types_idx}(i,:,:));
                curr_FR=mod_FR{cell_types_idx}{i};
                current_MIplotdata=mod_spectrum{cell_types_idx}{i};
                % draw raster
                for k=1:2
                    % draw psth ------------
                    sb1=subplot(2,3,2+(k-1));
                    hold on;
                    psth_observed=curr_FR(:,DIR==curr_pDIR,k);
                    psth_observed=psth_observed./(max(psth_observed(:)));
                    current_label=cell_types_idx;
                    % set color and tag to use
                    if current_label==2
                        coltuse=[255,150,0]./255;
                    elseif current_label==1
                        coltuse=[50,200,0]./255;
                    elseif current_label==3
                        coltuse=[150,150,150]./255;
                    end
                    hold on;
                    % draw psth
                    pipi=plot(gca,(0:length(psth_observed)).*(1/sr),[psth_observed(1);psth_observed],'-','Color',coltuse./(k),'LineWidth',2.5);
                    plot_shaded_auc(gca,(0:length(psth_observed))*(1/sr),[psth_observed(1);psth_observed]',0.15,coltuse./(k))
                    xlim([-0.2,1.2]);
                    ylim([-0,4]);
                    if k==1
                        tt=text(0.05,3.4,['pDIR grating = ',num2str(curr_pDIR),' d'],'FontSize',12);
                        ttt=text(0.05,3.2,['spike count grating = ','??'],'FontSize',12);
                    else
                        tt2=text(0.05,3.4,['pDIR plaid = ','??',' d'],'FontSize',12);
                        ttt2=text(0.05,3.2,['spike count plaid = ','??'],'FontSize',12);
                    end
                    plot([0,0],[0,5],'--k', 'LineWidth',2)
                    plot([1,1],[0,5],'--k', 'LineWidth',2)
                    hlabelx=get(gca,'Xlabel');
                    set(hlabelx,'String','time (s)','FontSize',12,'color','k')
                    hlabely=get(gca,'Ylabel');
                    set(hlabely,'String','normalized firing rate','FontSize',12,'color','k')
                    legend(gca,pipi,{'grating','plaid'})
                    title(['raster and psth (n=',num2str(i),') - DIR=',num2str(curr_pDIR),' - TF=','best',' - SF=','best'])
                    set(gca,'FontSize',12);
                end
                % plot polar plot tuning curve ----------------------------
                for k=1:2
                    if not(isempty(current_MIplotdata{curr_pDIR==DIR,k}))
                        subplot(2,3,5+(k-1));
                        % fetch data
                        N_f=current_MIplotdata{curr_pDIR==DIR,k}.N_f;
                        f=current_MIplotdata{curr_pDIR==DIR,k}.f;
                        fidx=current_MIplotdata{curr_pDIR==DIR,k}.fidx;
                        pow=current_MIplotdata{curr_pDIR==DIR,k}.pow;
                        meanspect=current_MIplotdata{curr_pDIR==DIR,k}.meanspect;
                        sigspect=current_MIplotdata{curr_pDIR==DIR,k}.sigspect;
                        tTF=current_MIplotdata{curr_pDIR==DIR,k}.TF;
                        F1F0=current_MIplotdata{curr_pDIR==DIR,k}.F1F0;
                        F1z=current_MIplotdata{curr_pDIR==DIR,k}.F1z;
                        % draw spectrum
                        plot(f(1:round(N_f/2)+1),pow(1:round(N_f/2)+1),'color',coltuse./(k),'LineWidth',3);
                        hold on
                        plot(f(fidx),pow(fidx),'o','LineWidth',6,'color',0*coltuse./(k));
                        plot(f(1:round(N_f/2)+1),(meanspect+sigspect)*ones(size(pow(1:round(N_f/2)+1))),'--','color',[0.5,0.5,0.5],'LineWidth',1);
                        plot(f(1:round(N_f/2)+1),(meanspect-sigspect)*ones(size(pow(1:round(N_f/2)+1))),'--','color',[0.5,0.5,0.5],'LineWidth',1);
                        plot(f(1:round(N_f/2)+1),(meanspect)*ones(size(pow(1:round(N_f/2)+1))),'.-','color',[0.5,0.5,0.5],'LineWidth',0.5);
                        ylimit=get(gca,'ylim');
                        xlimit=get(gca,'xlim');
                        ttt=text(0.5*xlimit(2),0.85*ylimit(2),['target TF = ',num2str(tTF),' Hz'],'FontSize',12);
                        set(gca,'FontSize',10);
                        hlabelx=get(gca,'Xlabel');
                        set(hlabelx,'String','f [Hz]','FontSize',10,'color','k')
                        hlabely=get(gca,'Ylabel');
                        set(hlabely,'String','PSD','FontSize',10,'color','k')
                        title(['Power spectrum (F1z = ',num2str(F1z),', F1F0 = ',num2str(F1F0),')']);
                        hold off
                        axis square
                        set(gca,'fontsize',12);
                    end
                end
                % plot polar plot tuning curve ----------------------------
                ppol=polaraxes('Position',[-0.05,0.25,.5,.5]);
                hold on;
                for k=1:2
                    temp_obs_tc=squeeze(curr_COUNT(:,k))'./max(squeeze(curr_COUNT(:,k)));
                    obs_tc=[temp_obs_tc,temp_obs_tc(1)];
                    % draw plar plots
                    p2=polarplot(ppol,[deg2rad(DIR),2*pi],obs_tc,'-');
                    set(p2,'color',coltuse./k)
                    set(p2, 'linewidth', 3.5);
                    % draw polarscatter of pref dir
                    ps2=polarscatter(ppol,deg2rad(curr_pDIR),obs_tc(find(DIR==curr_pDIR)),...
                        125,'markerfacecolor',coltuse./k,'markeredgecolor',coltuse./k);
                end
                title(ppol,['tuning polar plots (',cell_types_names{cell_types_idx},...
                    '-n=',num2str(i),')',...
                    ' - TF=','best',' - SF=','best'])
                set(ppol,'fontsize',12);
                sgtitle(['MI diagnostics ',cell_types_names{cell_types_idx},' n',num2str(i)])
                % save figure
                saveas(ff,[resultpath,filesep,'MI_diagnostics_',cell_types_names{cell_types_idx},'_n',num2str(i)],'jpg')
                print(ff,'-depsc','-painters',[[resultpath,filesep,'MI_diagnostics_',cell_types_names{cell_types_idx},'_n',num2str(i)],'.eps'])
                close all
            end
            % output advancement message
            fprintf(['neuron#',num2str(i),' class#',num2str(cell_types_idx),' analyzed...\n'])
        end
    end
    % save results
    save([resultpath,filesep,'MI_results_','DorsalNet','.mat'],...
        'mod_psth_per_type',...
        'mod_count_per_type',...
        'mod_nidx_per_type',...
        'mod_MIf1f0',...
        'mod_MIf1z',...
        'mod_spectrum',...
        'mod_FR',...
        'mod_pDIR');
else
    % load results
    load([resultpath,filesep,'MI_results_','DorsalNet','.mat'])
end

% initalize output datastructures - all
F1z_distr_grating=cell(1,numel(cell_types_codes));
F1z_distr_plaid=cell(1,numel(cell_types_codes));
PI_distr=cell(1,numel(cell_types_codes));
% initalize output datastructures - per layer
F1z_distr_grating_per_layer=cell(numel(layer_names),numel(cell_types_codes));
F1z_distr_plaid_per_layer=cell(numel(layer_names),numel(cell_types_codes));
PI_distr_per_layer=cell(numel(layer_names),numel(cell_types_codes));
% loop over cell classes
for cell_types_idx=1:numel(cell_types_codes)
    % initialize plotting datastructure - all
    F1z_distr_grating{cell_types_idx}=NaN(1,numel(mod_MIf1z{cell_types_idx}));
    F1z_distr_plaid{cell_types_idx}=NaN(1,numel(mod_MIf1z{cell_types_idx}));
    PI_distr{cell_types_idx}=NaN(1,numel(mod_MIf1z{cell_types_idx}));
    % loop over neurons
    for i=1:numel(mod_MIf1z{cell_types_idx})
        % get MI index
        F1z_distr_grating{cell_types_idx}(i)=mod_MIf1z{cell_types_idx}{i}(mod_pDIR{cell_types_idx}{i}(1)==DIR,1);
        F1z_distr_plaid{cell_types_idx}(i)=mod_MIf1z{cell_types_idx}{i}(mod_pDIR{cell_types_idx}{i}(1)==DIR,2);
    end
    % get pattern index
    PI_distr{cell_types_idx}=mod_Zp_per_type{cell_types_idx}-mod_Zc_per_type{cell_types_idx};
    % loop over layers - per layer
    for current_layer_id=1:numel(layer_names)
        % get current layer idx
        current_layer_idx=find(mod_layerlabel_per_type{cell_types_idx}==current_layer_id);
        % initialize plotting datastructure - per layer
        F1z_distr_grating_per_layer{current_layer_id,cell_types_idx}=NaN(1,numel(current_layer_idx));
        F1z_distr_plaid_per_layer{current_layer_id,cell_types_idx}=NaN(1,numel(current_layer_idx));
        PI_distr_per_layer{current_layer_id,cell_types_idx}=NaN(1,numel(current_layer_idx));
        % select current index values
        current_mod_pDIR=mod_pDIR{cell_types_idx}(current_layer_idx);
        current_mod_MIf1z=mod_MIf1z{cell_types_idx}(current_layer_idx);
        current_mod_Zp=mod_Zp_per_type{cell_types_idx}(current_layer_idx);
        current_mod_Zc=mod_Zc_per_type{cell_types_idx}(current_layer_idx);
        % loop over neurons
        for i=1:numel(current_mod_MIf1z)
            % get MI index
            F1z_distr_grating_per_layer{current_layer_id,cell_types_idx}(i)=current_mod_MIf1z{i}(current_mod_pDIR{i}(1)==DIR,1);
            F1z_distr_plaid_per_layer{current_layer_id,cell_types_idx}(i)=current_mod_MIf1z{i}(current_mod_pDIR{i}(1)==DIR,2);
        end
        % get pattern index
        PI_distr_per_layer{current_layer_id,cell_types_idx}=current_mod_Zp-current_mod_Zc;
    end
end

% set layer idx to exclude from analysis (last one after skip)
layer_idx_to_exclude=7;

% loop over cell classes
simple_frac=NaN(1,numel(cell_types_codes));
simple_num=NaN(1,numel(cell_types_codes));
complex_num=NaN(1,numel(cell_types_codes));
for cell_types_idx=1:numel(cell_types_codes)
    % count simple and complex cells
    n_simple=sum(F1z_distr_grating{cell_types_idx}(not(mod_layerlabel_per_type{cell_types_idx}==layer_idx_to_exclude))>3);
    n_complex=sum(F1z_distr_grating{cell_types_idx}(not(mod_layerlabel_per_type{cell_types_idx}==layer_idx_to_exclude))<=3);
    n_tot=n_simple+n_complex;
    simple_frac(cell_types_idx)=n_simple./n_tot;
    simple_num(cell_types_idx)=n_simple;
    complex_num(cell_types_idx)=n_complex;
end
% run Fisher exact test
x = table(simple_num(1:2)',complex_num(1:2)','VariableNames',{'Simple','Complex'},'RowNames',{'Component','Pattern'});
[h,pfishexact] = fishertest(x);

% initialize count component/pattern simple/complex structures 
fracs_comp=NaN(numel(layer_names),3);
fracs_patt=NaN(numel(layer_names),3);
fracs_uncl=NaN(numel(layer_names),3);
ns_comp=NaN(numel(layer_names),3);
ns_patt=NaN(numel(layer_names),3);
ns_uncl=NaN(numel(layer_names),3);
ns_tot=NaN(numel(layer_names),1);
for current_layer_id=1:numel(layer_names)
    % get simple/complex fractions for components
    n_comp_simple=sum(F1z_distr_grating_per_layer{current_layer_id,1}>=3);
    n_comp_complex=sum(F1z_distr_grating_per_layer{current_layer_id,1}<3);
    n_comp_tot=n_comp_simple+n_comp_complex;
    n_tot=numel(ctlabel_Os.(layer_names{current_layer_id}));
    fracs_comp(current_layer_id,:)=[n_comp_simple,n_comp_complex,n_comp_tot]./n_tot;
    ns_comp(current_layer_id,:)=[n_comp_simple,n_comp_complex,n_comp_tot];
    % get simple/complex fractions for patterns
    n_patt_simple=sum(F1z_distr_grating_per_layer{current_layer_id,2}>=3);
    n_patt_complex=sum(F1z_distr_grating_per_layer{current_layer_id,2}<3);
    n_patt_tot=n_patt_simple+n_patt_complex;
    n_tot=numel(ctlabel_Os.(layer_names{current_layer_id}));
    fracs_patt(current_layer_id,:)=[n_patt_simple,n_patt_complex,n_patt_tot]./n_tot;
    ns_patt(current_layer_id,:)=[n_patt_simple,n_patt_complex,n_patt_tot];
    % get simple/complex fractions for unclassified
    n_uncl_simple=sum(F1z_distr_grating_per_layer{current_layer_id,3}>=3);
    n_uncl_complex=sum(F1z_distr_grating_per_layer{current_layer_id,3}<3);
    n_uncl_tot=n_uncl_simple+n_uncl_complex;
    n_tot=numel(ctlabel_Os.(layer_names{current_layer_id}));
    fracs_uncl(current_layer_id,:)=[n_uncl_simple,n_uncl_complex,n_uncl_tot]./n_tot;
    ns_uncl(current_layer_id,:)=[n_uncl_simple,n_uncl_complex,n_uncl_tot];
    ns_tot(current_layer_id)=n_tot;
end

% get fraction of simple (over total of each component) vector
n_tot_simp=(ns_comp+ns_patt+ns_uncl);
n_tot_compl=ns_tot-n_tot_simp;
n_tot_simp=n_tot_simp(:,1);
n_tot_compl=n_tot_compl(:,2);
temp=(ns_comp+ns_patt+ns_uncl)./ns_tot;
overall_fraction_simple=temp(:,1);
temp=(ns_comp)./n_tot_simp;
comp_fraction_simple=temp(:,1);
temp=(ns_patt)./n_tot_simp;
patt_fraction_simple=temp(:,1);
temp=(ns_comp)./n_tot_compl;
comp_fraction_compl=temp(:,2);
temp=(ns_patt)./n_tot_compl;
patt_fraction_compl=temp(:,2);
n_tot_comp=ns_comp(:,3);
simp_fraction_comp=ns_comp(:,1)./n_tot_comp;
n_tot_patt=ns_patt(:,3);
simp_fraction_patt=ns_patt(:,1)./n_tot_patt;
n_tot_uncl=ns_uncl(:,3);
simp_fraction_uncl=ns_uncl(:,1)./n_tot_uncl;

% get fraction of component and simple (over total pattern + component) vector
n_tot_nuncl=ns_comp+ns_patt;
n_tot_nuncl=n_tot_nuncl(:,3);
comp_franction_nuncl=ns_comp(:,3)./n_tot_nuncl;
simp_franction_nuncl=(ns_comp(:,1)+ns_patt(:,1))./n_tot_nuncl;

% plot MI analysis result (violins) ---------------------------------------
fighand5=figure('units','normalized','outerposition',[0 0 1 1]);
% plot distributions of MI --- (gratings)
subplot(1,3,1)
% decide wheter to use max or unrolled
clear distribtouse
distribtouse{1}=F1z_distr_grating{1}(not(mod_layerlabel_per_type{1}==layer_idx_to_exclude));
distribtouse{2}=F1z_distr_grating{2}(not(mod_layerlabel_per_type{2}==layer_idx_to_exclude));
inputpars.inputaxh=gca;
hold(inputpars.inputaxh,'on')
% set settings for violin distribution plotting
inputpars.boxplotwidth=0.4;%0.5;
inputpars.boxplotlinewidth=2;
inputpars.densityplotwidth=0.5;%0.5;
inputpars.yimtouse=[-0.5,5];
% inputpars.yimtouse=[0,8];
inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
inputpars.scatteralpha=0.15;
inputpars.scattersize=20;
inputpars.distralpha=0.5;
inputpars.xlabelstring=[];
inputpars.ylabelstring='MI';
inputpars.titlestring=['modulation index (','MI',') - ','gratings',' ( comp # = ',...
    num2str(numel(distribtouse{1}),'%.0f'),...
    ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),')'];
inputpars.boolscatteron=1;
inputpars.ks_bandwidth=0.205;
inputpars.xlimtouse=[-0.5,3.5]; %[-1,5];
% plot violins
inputadata.inputdistrs=distribtouse;
inputpars.n_distribs=numel(inputadata.inputdistrs);
inputpars.dirstrcenters=(1:inputpars.n_distribs);
inputpars.xtickslabelvector={'component','pattern'};
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputaxh = plot_violinplot_PN_new(inputadata,inputpars); %#ok<NASGU>
hold on;
plot(get(gca,'xlim'),[3,3],'--','linewidth',2,'color',[0.5,0.5,0.5])
[pval_csi_pc,~] = ranksum(distribtouse{1}',distribtouse{2}');
[~,pval_mi_pc_mu] = ttest2(distribtouse{1}',distribtouse{2}');
median_MI_comp=nanmedian(distribtouse{1});
median_MI_patt=nanmedian(distribtouse{2});
usedxlim=get(gca,'xlim'); %#ok<NASGU>
hold on;
text(gca,0,4.25,['comp vs. patt median mi diff p = ',num2str(pval_csi_pc)],'fontsize',12)
text(gca,0,4,['comp vs. patt mean mi diff p = ',num2str(pval_mi_pc_mu)],'fontsize',12)
text(gca,0,4.5,['comp median val = ',num2str(median_MI_comp)],'fontsize',12)
text(gca,0,4.75,['patt median val = ',num2str(median_MI_patt)],'fontsize',12)
xtickangle(45)
set(gca,'fontsize',12)
axis square
% plot distributions of MI --- (plaids)
subplot(1,3,2)
% decide wheter to use max or unrolled
clear distribtouse
distribtouse{1}=F1z_distr_plaid{1}(not(mod_layerlabel_per_type{1}==layer_idx_to_exclude));
distribtouse{2}=F1z_distr_plaid{2}(not(mod_layerlabel_per_type{2}==layer_idx_to_exclude));
inputpars.inputaxh=gca;
hold(inputpars.inputaxh,'on')
% set settings for violin distribution plotting
inputpars.boxplotwidth=0.4;%0.5;
inputpars.boxplotlinewidth=2;
inputpars.densityplotwidth=0.5;%0.5;
inputpars.yimtouse=[-0.5,5];
% inputpars.yimtouse=[0,8];
inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
inputpars.scatteralpha=0.15;
inputpars.scattersize=20;
inputpars.distralpha=0.5;
inputpars.xlabelstring=[];
inputpars.ylabelstring='MI';
inputpars.titlestring=['modulation index (','MI',') - ','gratings',' ( comp # = ',...
    num2str(numel(distribtouse{1}),'%.0f'),...
    ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),')'];
inputpars.boolscatteron=1;
inputpars.ks_bandwidth=0.205;
inputpars.xlimtouse=[-0.5,3.5]; %[-1,5];
% plot violins
inputadata.inputdistrs=distribtouse;
inputpars.n_distribs=numel(inputadata.inputdistrs);
inputpars.dirstrcenters=(1:inputpars.n_distribs);
inputpars.xtickslabelvector={'component','pattern'};
inputpars.distrcolors{1}=([50,200,0]./255)/2;
inputpars.distrcolors{2}=([255,150,0]./255)/2;
inputaxh = plot_violinplot_PN_new(inputadata,inputpars);
hold on;
plot(get(gca,'xlim'),[3,3],'--','linewidth',2,'color',[0.5,0.5,0.5])
[pval_mi_pc,~] = ranksum(distribtouse{1}',distribtouse{2}');
[~,pval_mi_pc_mu] = ttest2(distribtouse{1}',distribtouse{2}');
median_MI_comp=nanmedian(distribtouse{1});
median_MI_patt=nanmedian(distribtouse{2});
usedxlim=get(gca,'xlim');
hold on;
text(gca,0,4.25,['comp vs. patt median mi diff p = ',num2str(pval_mi_pc)],'fontsize',12)
text(gca,0,4,['comp vs. patt mean mi diff p = ',num2str(pval_mi_pc_mu)],'fontsize',12)
text(gca,0,4.5,['comp median val = ',num2str(median_MI_comp)],'fontsize',12)
text(gca,0,4.75,['patt median val = ',num2str(median_MI_patt)],'fontsize',12)
xtickangle(45)
set(gca,'fontsize',12)
axis square
% plot simple/complex barplot --- (gratings)
subplot(1,3,3)
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
totnum=simple_num+complex_num;
simplefrac=simple_num./totnum;
barposition=[1,2];
hold on;
bar(barposition(1),simplefrac(1),'facecolor',inputpars.distrcolors{1},...
    'edgecolor',inputpars.distrcolors{1})
bar(barposition(2),simplefrac(2),'facecolor',inputpars.distrcolors{2},...
    'edgecolor',inputpars.distrcolors{2})
ylabel('fraction')
xticks(barposition)
xticklabels({'comp-simp', 'patt-simp'})
hold on;
xtickangle(45)
ylabel('fraction')
set(gca,'fontsize',12)
xlim([0,3])
ylim([0,0.7])
axis square
ylimused=get(gca,'ylim');
text(gca,1.6,0.9.*ylimused(end),['# comp-simp = ',num2str(simple_num(1))],'fontsize',12)
text(gca,1.6,0.86.*ylimused(end),['# patt-simp = ',num2str(simple_num(2))],'fontsize',12)
text(gca,1.6,0.82.*ylimused(end),['p fisher exact = ',num2str(round(pfishexact,4))],'fontsize',12)
title(['fraction of simple cells - ','gratings',' ( comp # = ',...
    num2str(numel(distribtouse{1}),'%.0f'),...
    ' - patt #  = ',num2str(numel(distribtouse{2}),'%.0f'),')']);
% add suptitile
sgtitle('grating and plaid MI distributions (pattern vs. components) - DorsalNet')
saveas(fighand5,[resultpath,filesep,'MI_distributions_DorsalNet'],'jpg')
print(fighand5,'-depsc','-painters',[resultpath,filesep,'MI_distributions_DorsalNet','.eps'])
close all

% plot MI analysis result (PI scatter) ------------------------------------
fighand6=figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=0.5*[255,255,255]./255;
scatter(PI_distr{3}(not(mod_layerlabel_per_type{3}==layer_idx_to_exclude)),F1z_distr_grating{3}(not(mod_layerlabel_per_type{3}==layer_idx_to_exclude)),...
    100,'MarkerFaceColor',inputpars.distrcolors{3},'MarkerEdgeColor',inputpars.distrcolors{3})
scatter(PI_distr{1}(not(mod_layerlabel_per_type{1}==layer_idx_to_exclude)),F1z_distr_grating{1}(not(mod_layerlabel_per_type{1}==layer_idx_to_exclude)),...
    100,'MarkerFaceColor',inputpars.distrcolors{1},'MarkerEdgeColor',inputpars.distrcolors{1})
scatter(PI_distr{2}(not(mod_layerlabel_per_type{2}==layer_idx_to_exclude)),F1z_distr_grating{2}(not(mod_layerlabel_per_type{2}==layer_idx_to_exclude)),...
    100,'MarkerFaceColor',inputpars.distrcolors{2},'MarkerEdgeColor',inputpars.distrcolors{2})
plot([0,0],get(gca,'ylim'),'--','linewidth',2,'color',[0.5,0.5,0.5])
ylabel('MI')
xlabel('PI')
PI_distr_all=[PI_distr{1}(not(mod_layerlabel_per_type{1}==layer_idx_to_exclude))',PI_distr{2}(not(mod_layerlabel_per_type{2}==layer_idx_to_exclude))',PI_distr{3}(not(mod_layerlabel_per_type{3}==layer_idx_to_exclude))']';
F1z_distr_grating_all=[F1z_distr_grating{1}(not(mod_layerlabel_per_type{1}==layer_idx_to_exclude)),F1z_distr_grating{2}(not(mod_layerlabel_per_type{2}==layer_idx_to_exclude)),F1z_distr_grating{3}(not(mod_layerlabel_per_type{3}==layer_idx_to_exclude))]';
valid_idx=find(not(isnan(F1z_distr_grating_all)));
[coor_v,corr_p]=corr(PI_distr_all(valid_idx),F1z_distr_grating_all(valid_idx));
text(gca,-15,3.75,['PI vs. MI corr v = ',num2str(round(coor_v,2))],'fontsize',12)
text(gca,-15,3.65,['PI vs. MI corr p = ',num2str(corr_p)],'fontsize',12)
xlim([-20,20])
title(' modulation index vs. pattern index scatter (pattern vs. components) - gratings - DorsalNet')
set(gca,'fontsize',12)
axis square
saveas(fighand6,[resultpath,filesep,'MI_PI_scatter_DorsalNet'],'jpg')
print(fighand6,'-depsc','-painters',[resultpath,filesep,'MI_PI_scatter_DorsalNet','.eps'])
close all

% plot MI analysis result (violins) per layer -----------------------------
fighand7=figure('units','normalized','outerposition',[0 0 1 1]);
ccolors{1}=[50,200,0]./255;
ccolors{2}=[255,150,0]./255;
ccolors{3}=0.5*[255,255,255]./255;
for cell_types_idx=1:numel(cell_types_codes)
% decide wheter to use max or unrolled
distribtouse=F1z_distr_grating_per_layer(:,cell_types_idx);
inputpars.inputaxh=subplot(1,3,1);
hold(inputpars.inputaxh,'on')
% set settings for violin distribution plotting
inputpars.boxplotwidth=0.4;%0.5;
inputpars.boxplotlinewidth=2;
inputpars.densityplotwidth=0.5;%0.5;
inputpars.yimtouse=[0,5];
% inputpars.yimtouse=[0,8];
inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
inputpars.scatteralpha=0.15;
inputpars.scattersize=20;
inputpars.distralpha=0.5;
inputpars.xlabelstring=[];
inputpars.ylabelstring='MI';
inputpars.titlestring=['modulation index (','MI',') - ','gratings'];
inputpars.boolscatteron=1;
inputpars.ks_bandwidth=0.15;
% plot violins
inputadata.inputdistrs=distribtouse;
inputpars.n_distribs=numel(inputadata.inputdistrs);
inputpars.dirstrcenters=(1:4:(4*numel(distribtouse)))+(cell_types_idx-1);
inputpars.xlimtouse=[min(inputpars.dirstrcenters)-2,max(inputpars.dirstrcenters)+2]; %[-1,5];
inputpars.xtickslabelvector=layer_names;
current_color=ccolors{cell_types_idx};
inputpars.distrcolors{1}=current_color;
inputpars.distrcolors{2}=current_color;
inputpars.distrcolors{3}=current_color;
inputpars.distrcolors{4}=current_color;
inputpars.distrcolors{5}=current_color;
inputpars.distrcolors{6}=current_color;
inputpars.distrcolors{7}=current_color;
inputaxh = plot_violinplot_PN_new(inputadata,inputpars);
hold on;
end
plot(get(gca,'xlim'),[3,3],'--','linewidth',2,'color',[0.5,0.5,0.5])
axis square
subplot(1,3,2)
hold on;
xvals1=(1:4:(4*numel(distribtouse)))+(2-1);
bar(xvals1',comp_fraction_simple,'facecolor',ccolors{1},'edgecolor',ccolors{1},'barwidth',0.25);
xvals2=(1:4:(4*numel(distribtouse)))+(3-1);
bar(xvals2',patt_fraction_simple,'facecolor',ccolors{2},'edgecolor',ccolors{2},'barwidth',0.25);
xvals3=(1:4:(4*numel(distribtouse)))+(1-1);
bar(xvals3,overall_fraction_simple,'facecolor',0*ccolors{3},'edgecolor',0*ccolors{3},'barwidth',0.25);
ylabel('fraction')
xticks(xvals1)
xticklabels(layer_names)
xtickangle(45)
set(gca,'fontsize',12)
title('fraction of simple cells and pattern/component subfraction')
axis square
subplot(1,3,3)
hold on;
xvals1=(1:4:(4*numel(distribtouse)))+(2-1);
bar(xvals1',comp_fraction_compl,'facecolor',ccolors{1},'edgecolor',ccolors{1},'barwidth',0.25);
xvals2=(1:4:(4*numel(distribtouse)))+(3-1);
bar(xvals2',patt_fraction_compl,'facecolor',ccolors{2},'edgecolor',ccolors{2},'barwidth',0.25);
xvals3=(1:4:(4*numel(distribtouse)))+(1-1);
bar(xvals3,1-overall_fraction_simple,'facecolor',0*ccolors{3},'edgecolor',0*ccolors{3},'barwidth',0.25);
ylabel('fraction')
xticks(xvals1)
xticklabels(layer_names)
xtickangle(45)
set(gca,'fontsize',12)
title('fraction of complex cells and pattern/component subfraction')
axis square
sgtitle('patternness/componentness vs. complexness/simpleness across layers of DorsalNet')
saveas(fighand7,[resultpath,filesep,'pattcomp_complsimpl_bylayer_DorsalNet'],'jpg')
print(fighand7,'-depsc','-painters',[resultpath,filesep,'pattcomp_complsimpl_bylayer_DorsalNet','.eps'])
close all

% plot fraction of simple among patterna and components -------------------
fighand8=figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
xvals1=(1:4:(4*numel(simp_fraction_comp)))+(1-1);
bar(xvals1',simp_fraction_comp,'facecolor',ccolors{1},'edgecolor',ccolors{1},'barwidth',0.25);
for jj=1:numel(simp_fraction_comp)
  text(xvals1(jj)-0.5,simp_fraction_comp(jj)+0.01,['n=',num2str(n_tot_comp(jj))],'color',ccolors{1},'fontweight','bold','fontsize',9)  
end
xvals2=(1:4:(4*numel(simp_fraction_patt)))+(2-1);
bar(xvals2',simp_fraction_patt,'facecolor',ccolors{2},'edgecolor',ccolors{2},'barwidth',0.25);
for jj=1:numel(simp_fraction_patt)
  text(xvals2(jj)-0.5,simp_fraction_patt(jj)+0.01,['n=',num2str(n_tot_patt(jj))],'color',ccolors{2},'fontweight','bold','fontsize',9)  
end
xvals3=(1:4:(4*numel(simp_fraction_uncl)))+(3-1);
bar(xvals3,simp_fraction_uncl,'facecolor',ccolors{3},'edgecolor',ccolors{3},'barwidth',0.25);
for jj=1:numel(simp_fraction_uncl)
  text(xvals3(jj)-0.5,simp_fraction_uncl(jj)+0.01,['n=',num2str(n_tot_uncl(jj))],'color',ccolors{3},'fontweight','bold','fontsize',9)  
end
ylabel('simple fraction')
xticks(xvals2)
xticklabels(layer_names)
xtickangle(45)
ylim([0,1.1])
set(gca,'fontsize',12)
title('fraction of simple cells per cell type and layer')
axis square
saveas(fighand8,[resultpath,filesep,'pattcomp_complsimpl_bylayer_DorsalNet_bis'],'jpg')
print(fighand8,'-depsc','-painters',[resultpath,filesep,'pattcomp_complsimpl_bylayer_DorsalNet_bis','.eps'])
close all

% plot fraction of simple and components over total and pattern and components (not unclassified) -----
fighand9=figure('units','normalized','outerposition',[0 0 1 1]);
hold on;
xvals1=(1:3:(3*numel(comp_franction_nuncl)))+(1-1);
bar(xvals1',comp_franction_nuncl,'facecolor',0*ccolors{1},'edgecolor',0*ccolors{1},'barwidth',0.25);
for jj=1:numel(comp_franction_nuncl)
  text(xvals1(jj)-0.35,comp_franction_nuncl(jj)+0.01,['n=',num2str(n_tot_nuncl(jj))],'color',0*ccolors{1},'fontweight','bold','fontsize',9)  
end
plot(xvals1(1:end-1),comp_franction_nuncl(1:end-1),'--','color',0*ccolors{1},'linewidth',2)
xvals2=(1:3:(3*numel(simp_franction_nuncl)))+(2-1);
bar(xvals2',simp_franction_nuncl,'facecolor',ccolors{1},'edgecolor',ccolors{1},'barwidth',0.25);
for jj=1:numel(simp_franction_nuncl)
  text(xvals2(jj)-0.35,simp_franction_nuncl(jj)+0.01,['n=',num2str(n_tot_nuncl(jj))],'color',ccolors{1},'fontweight','bold','fontsize',9)  
end
plot(xvals2(1:end-1),simp_franction_nuncl(1:end-1),'--','color',ccolors{1},'linewidth',2)
ylabel('fraction')
xticks(xvals1+0.5)
xticklabels(layer_names)
xtickangle(45)
ylim([0,1.1])
xlim([0,21])
set(gca,'fontsize',12)
title('fraction of simple / component cells (over total non-unclassified)')
axis square
saveas(fighand9,[resultpath,filesep,'simp_comp_bylayer_DorsalNet'],'jpg')
print(fighand9,'-depsc','-painters',[resultpath,filesep,'simp_comp_bylayer_DorsalNet','.eps'])
close all