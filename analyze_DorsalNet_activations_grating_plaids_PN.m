clear all
close all
clc

%% load data -------------------------------------------------

% set paths
codepath='E:\Backups\Personal_bk\DorsalNet\code';
oldcode='E:\Backups\Personal_bk\DorsalNet\acute_analysis_code';
datapath='E:\Backups\Personal_bk\DorsalNet\activations';
resultpath='E:\Backups\Personal_bk\DorsalNet\results';
% add paths
addpath(genpath(codepath));
addpath(genpath(oldcode));
addpath(datapath)
addpath(resultpath)

% load grating activations
Grating_activations=load([datapath,filesep,'Grating_responses_Matlab3bis.mat']);
% load grating activations
Plaid_activations=load([datapath,filesep,'Plaid_responses_Matlab3bis.mat']);
% load grating activations
Example_noise_activations=load([datapath,filesep,'Noise_responses0_Matlab2.mat']);

% get layer names
layer_names=fieldnames(Grating_activations);

% get stimulus types
stimulus_types={'gratings','plaids'};
cell_types_codes=[1,2,0];
cell_types_names={'unclassified','component','pattern'};

% get grating and plaid length
grating_and_plaid_length=size(Grating_activations.(layer_names{1}),3);

% set pars
pars = set_pars_PN();
DIR=pars.stimPars.DIR;
SR=1/pars.stimPars.frame_duration;
cell_type_codes=[1,2,0];

% set whether to produce single neuron plots or not
plotsingleneubool=0;

%% preprocess activations (gratings and plaids) -------------------------------------------------

% initialize arrays of tuning curves and psths
PSTH_observed=cell2struct(cell(size(layer_names)),layer_names);
TCu_observed=cell2struct(cell(size(layer_names)),layer_names);
TC_observed=cell2struct(cell(size(layer_names)),layer_names);
Zp_observed=cell2struct(cell(size(layer_names)),layer_names);
Zc_observed=cell2struct(cell(size(layer_names)),layer_names);
Clabel_observed=cell2struct(cell(size(layer_names)),layer_names);
Rlabel_observed=cell2struct(cell(size(layer_names)),layer_names);
DSI_observed=cell2struct(cell(size(layer_names)),layer_names);
OSI_observed=cell2struct(cell(size(layer_names)),layer_names);
pDIR_observed=cell2struct(cell(size(layer_names)),layer_names);
Lid_observed=cell2struct(cell(size(layer_names)),layer_names);

% loop over layers
for current_layer_id=1:numel(layer_names)
    
    tic
    
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % get current layer neuron number
    current_layer_nn=size(Grating_activations.(current_layer_name),2);
    % initialize data storage matrices for current layer
    PSTH_observed.(current_layer_name)=NaN(current_layer_nn,grating_and_plaid_length,numel(stimulus_types));
    TCu_observed.(current_layer_name)=NaN(current_layer_nn,numel(DIR),numel(stimulus_types));
    TC_observed.(current_layer_name)=NaN(current_layer_nn,numel(DIR),numel(stimulus_types));
    Zp_observed.(current_layer_name)=NaN(current_layer_nn,1);
    Zc_observed.(current_layer_name)=NaN(current_layer_nn,1);
    Clabel_observed.(current_layer_name)=NaN(current_layer_nn,1);
    Rlabel_observed.(current_layer_name)=NaN(current_layer_nn,numel(stimulus_types));
    DSI_observed.(current_layer_name)=NaN(current_layer_nn,numel(stimulus_types));
    OSI_observed.(current_layer_name)=NaN(current_layer_nn,numel(stimulus_types));
    pDIR_observed.(current_layer_name)=NaN(current_layer_nn,numel(stimulus_types));
    Lid_observed.(current_layer_name)=NaN(current_layer_nn,1);
    
    % loop over neurons
    for current_n=1:current_layer_nn
        
        % TODO: use all the noise for this, get out distributions of z
        % scores P and G per layer, select on the basis of that
        
        % get noise mu and std
        mu_noise_activation=max(nanmean(Example_noise_activations.(current_layer_name)(:,current_n,:)),eps);
        std_noise_activation=max(nanstd(Example_noise_activations.(current_layer_name)(:,current_n,:)),eps);
        
        % loop over stimulus types
        for current_stimulus_type_id=1:numel(stimulus_types)
            
            if current_stimulus_type_id==1 % grating case
                % select current neuron rectify and squeeze - grating
                current_activations=Grating_activations.(current_layer_name)(:,current_n,:);
                current_activations=(current_activations-mu_noise_activation)./std_noise_activation;
                tempmat=squeeze(max(...
                    current_activations,...
                    zeros(size(Grating_activations.(current_layer_name)(:,current_n,:)))...
                    ));
            elseif current_stimulus_type_id==2 % plaid case
                % select current neuron rectify and squeeze - plaid
                current_activations=Plaid_activations.(current_layer_name)(:,current_n,:);
                current_activations=(current_activations-mu_noise_activation)./std_noise_activation;
                tempmat=squeeze(max(...
                    current_activations,...
                    zeros(size(Plaid_activations.(current_layer_name)(:,current_n,:)))...
                    ));
            end
            
            % integrate to get tuning curve
            temp_tc_obs_raw=nansum(tempmat,2);
            % get unnormalized tuning curve (interpolating)
            temp_tc_un_observed=temp_tc_obs_raw'+eps*rand(1,numel(DIR))+100*eps; % interp1(1:16,squeeze(temp_tc_obs_raw),1:numel(DIR))+eps*rand(1,numel(DIR))+100*eps;
            % get normalized tuning curve
            temp_tc_observed=temp_tc_un_observed...
                ./(max(temp_tc_un_observed(:)));
            % store results
            TCu_observed.(current_layer_name)(current_n,:,current_stimulus_type_id)=temp_tc_un_observed;
            TC_observed.(current_layer_name)(current_n,:,current_stimulus_type_id)=temp_tc_observed;
            
            % store responsive / unresponsive label
            if sum(temp_tc_obs_raw==0)==length(isnan(temp_tc_observed))
                Rlabel_observed.(current_layer_name)(current_n,current_stimulus_type_id)=0;
            else
                Rlabel_observed.(current_layer_name)(current_n,current_stimulus_type_id)=1;
            end
            
            % get pDIR OSI and DSI
            [ temp_OSI,temp_DSI,~,~,temp_pDIR_idx  ] = compute_SIs( temp_tc_observed );
            temp_pDIR=DIR(temp_pDIR_idx);
            % store results
            DSI_observed.(current_layer_name)(current_n,current_stimulus_type_id)=temp_DSI;
            OSI_observed.(current_layer_name)(current_n,current_stimulus_type_id)=temp_OSI;
            pDIR_observed.(current_layer_name)(current_n,current_stimulus_type_id)=temp_pDIR;
            
            % fetch current grating preferred direction
            curr_grat_pDIR=pDIR_observed.(current_layer_name)(current_n,1);
            % get current grating preferred direction psth
            temp_psth_observed=tempmat(curr_grat_pDIR==DIR,:);
            temp_psth_observed=squeeze(temp_psth_observed./(max(temp_psth_observed(:))));
            % store results
            PSTH_observed.(current_layer_name)(current_n,:,current_stimulus_type_id)=temp_psth_observed;
            
        end
        
        % get current grating and plaid tc
        curr_grat_tc=TC_observed.(current_layer_name)(current_n,:,1);
        curr_plaid_tc=TC_observed.(current_layer_name)(current_n,:,2);
        % perform partial correlation analysis
        [ temp_PI, ~, temp_Zp, temp_Zc, ~, ~, ~, ~, ~, ~ ] =...
            get_pattern_index( curr_grat_tc,curr_plaid_tc );
        % store results
        Zp_observed.(current_layer_name)(current_n,1)=temp_Zp;
        Zc_observed.(current_layer_name)(current_n,1)=temp_Zc;
        Lid_observed.(current_layer_name)(current_n,1)=current_layer_id;
        
        % classify unit as pattern or component
        if temp_Zp-max(temp_Zc,0)>=1.28
            temp_label=2; % 2 -> pattern
        elseif temp_Zc-max(temp_Zp,0)>=1.28
            temp_label=1; % 1 -> component
        else
            temp_label=0; % 0 -> unclassified
        end
        % store results
        Clabel_observed.(current_layer_name)(current_n,1)=temp_label;
        
    end
    
    toc
    fprintf([current_layer_name,' grating and plaid responses analyzed ...\n'])
    
end

%% plot bullet plots -------------------------------------------------

% initialize figure
f1 = figure('units','normalized','outerposition',[0 0 1 1]);

% loop over layers
for current_layer_id=1:numel(layer_names)
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % get colors
    inputpars.distrcolors{3}=[0,0,0];
    inputpars.distrcolors{1}=[50,200,0]./255;
    inputpars.distrcolors{2}=[255,150,0]./255;
    % set subplot
    subplot(2,4,current_layer_id)
    hold on;
    % loop over neurons
    for current_n=1:numel(Clabel_observed.(current_layer_name))
        % fetch values
        Clabel=Clabel_observed.(current_layer_name)(current_n);
        Rlabel=sum(Rlabel_observed.(current_layer_name)(current_n,:));
        Zc=Zc_observed.(current_layer_name)(current_n);
        Zp=Zp_observed.(current_layer_name)(current_n);
        DSI=DSI_observed.(current_layer_name)(current_n,1);
        % if responsive
        if not(sum(Rlabel)==0)
            % get cell class idx
            cell_class_idx=find(Clabel==cell_type_codes);
            % plot in bullet
            scatter(Zc,Zp,75,...
                'MarkerFaceColor',inputpars.distrcolors{cell_class_idx},'MarkerEdgeColor',inputpars.distrcolors{cell_class_idx},...
                'MarkerFaceAlpha',0.25,'MarkerEdgeAlpha',0.25);
            if DSI>=0.33
                % plot in bullet as DS
                plot(Zc,Zp,'.','MarkerSize',15,'Color',inputpars.distrcolors{cell_class_idx});
            end
        end
    end
    line([0 8], [1.28 9.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    line([1.28 9.28], [0 8],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    line([1.28 1.28], [-5 0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    line([-5 0], [1.28 1.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
    xlim([-5,9])
    ylim([-5,9])
    axis square
    xlabel('Zc'); ylabel('Zp');
    title(['Zp vs. Zc ',current_layer_name,' ( ntot=',...
        num2str(numel(Clabel_observed.(current_layer_name))),' ntotDS='...
        ,num2str(sum(DSI_observed.(current_layer_name)(:,1)>=0.33)),' )']);
    set(gca,'fontsize',12)
end
suptitle('DorsalNet - bullet plot - pattern and components across layers')
% save plot
saveas(f1,[resultpath,filesep,'bullets_across_layers'],'jpg')
print(f1,'-depsc','-painters',[[resultpath,filesep,'bullets_across_layers'],'.eps'])

%% plot barplot pattern and component fraction -------------------------.

% set ds threshold
DSth=0;

% initialize reclassification count (i.e. category change) storage stuctures
class_count=cell2struct(cell(size(layer_names)),layer_names);
% loop over layers
for current_layer_id=1:numel(layer_names)
    % get current layer name
    current_layer_name=layer_names{current_layer_id};
    % get current labels
    current_labl=Clabel_observed.(current_layer_name);
    % keep only responsive
    current_rlabl=logical(sum(Rlabel_observed.(current_layer_name),2));  
    current_validlabl=and(current_rlabl,DSI_observed.(current_layer_name)(:,1)>=DSth);
    current_labl=current_labl(current_validlabl==1);
    class_count.(current_layer_name)=NaN(1,3);
    % count per layer
    class_count.(current_layer_name)(1)=sum(cell_types_codes(1)==current_labl);
    class_count.(current_layer_name)(2)=sum(cell_types_codes(2)==current_labl);
    class_count.(current_layer_name)(3)=sum(cell_types_codes(3)==current_labl);
end
% bult fraction matrices
count_mat=concatenate_layers(class_count);
frac_mat=count_mat./repmat(sum(count_mat,2),[1,3]);
rat_mat=count_mat(:,1:2)./repmat(sum(count_mat(:,1:2),2),[1,2]);

% initialize figure
fighand2tris = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1)
% get input for barplot
comp_fracs=frac_mat(:,1);
patt_fracs=frac_mat(:,2);
uncl_fracs=frac_mat(:,3);
comp_xtouse=1:7;
patt_xtouse=7+3+(1:7);
uncl_xtouse=14+6+(1:7);
%get colors
compcol=[50,200,0]./255;
pattcol=[255,150,0]./255;
unclcol=[255/2,255/2,255/2]./255;
% plot bars
hold on;
bar(comp_xtouse,comp_fracs,...
    'facecolor',compcol,...
    'edgecolor',compcol,...
    'facealpha',0.5,...
    'linewidth',1.5...
    ) %#ok<*NBRAK>
bar(patt_xtouse,patt_fracs,...
    'facecolor',pattcol,...
    'edgecolor',pattcol,...
    'facealpha',0.5,...
    'linewidth',1.5...
    ) %#ok<*NBRAK>
bar(uncl_xtouse,uncl_fracs,...
    'facecolor',unclcol,...
    'edgecolor',unclcol,...
    'facealpha',0.5,...
    'linewidth',1.5...
    ) %#ok<*NBRAK>
titlestring=['all layers - fraction per class '];
title(titlestring)
ylim([0,0.7])
xlim([-1,uncl_xtouse(end)+2])
% [chi2stat,p_chitest] = chiSquareTest([comp_n_dis',patt_n_dis']); 
% text(1,0.9,['chi square p = ',num2str(p_chitest)],'fontsize',12);
xtouse=[comp_xtouse,patt_xtouse,uncl_xtouse];
xticks(xtouse)
xtouselabls=[layer_names',layer_names',layer_names'];
xticklabels(xtouselabls)
xtickangle(45)
xlabel('')
ylabel('fraction of cells')
set(gca,'fontsize',12)
axis square
subplot(1,2,2)
% get input for barplot
comp_fracs=rat_mat(:,1);
patt_fracs=rat_mat(:,2);
comp_xtouse=1:7;
patt_xtouse=1:7;
%get colors
compcol=[50,200,0]./255;
pattcol=[255,150,0]./255;
% plot bars
hold on;
bar(patt_xtouse,patt_fracs,...
    'facecolor',(pattcol)./1.3,...
    'edgecolor',(pattcol)./1.3,...
    'facealpha',0.5,...
    'linewidth',1.5,...
    'barwidth',0.75) %#ok<*NBRAK>
% plot_shaded_distribution(gca,[0,patt_xtouse,patt_xtouse(end)+1],[0,patt_fracs',0],0.2,(pattcol))
% plot([0,patt_xtouse,patt_xtouse(end)+1],[0,patt_fracs',0],'linewidth',3,'color',(pattcol))
% plot_shaded_distribution(gca,[0,comp_xtouse,comp_xtouse(end)+1],[0,comp_fracs',0],0.2,(compcol)./1.5)
% plot([0,comp_xtouse,comp_xtouse(end)+1],[0,comp_fracs',0],'linewidth',3,'color',(compcol)./1.5)
titlestring=['all layers - pattern vs. component ratio '];
title(titlestring)
ylim([0,0.8])
xlim([-1,patt_xtouse(end)+2])
% [chi2stat,p_chitest] = chiSquareTest([comp_n_dis',patt_n_dis']); 
% text(1,0.9,['chi square p = ',num2str(p_chitest)],'fontsize',12);
xtouse=[patt_xtouse];
xticks(xtouse)
xtouselabls=[layer_names'];
xticklabels(xtouselabls)
xtickangle(45)
xlabel('')
ylabel('pattern vs. component ratio')
set(gca,'fontsize',12)
axis square
suptitle(['DorsalNet - cell class distributions across layers - DSth = ',num2str(DSth),' - ( # resp comp = ',...
    num2str(sum(count_mat(1,:),2),'%.0f'),...
    ' - # resp patt = ',num2str(sum(count_mat(2,:),2),'%.0f'),...
    ' - # resp uncl = ',num2str(sum(count_mat(3,:),2),'%.0f'),' )'])
% save plot
saveas(fighand2tris,[resultpath,filesep,'classification_barplots'],'jpg')
print(fighand2tris,'-depsc','-painters',[[resultpath,filesep,'classification_barplots'],'.eps'])

%% plot tuning curves  -------------------------------------------------

if plotsingleneubool
    
    % loop over layers
    for current_layer_id=1:numel(layer_names)
        % get current layer name
        current_layer_name=layer_names{current_layer_id};
        % loop over neurons
        for current_n=1:numel(Clabel_observed.(current_layer_name))
            % fetch values
            Clabel=Clabel_observed.(current_layer_name)(current_n);
            Rlabel=Rlabel_observed.(current_layer_name)(current_n,:);
            Zc=Zc_observed.(current_layer_name)(current_n);
            Zp=Zp_observed.(current_layer_name)(current_n);
            DSI=DSI_observed.(current_layer_name)(current_n,1);
            pDIR_grat=pDIR_observed.(current_layer_name)(current_n,1);
            pDIR_plaid=pDIR_observed.(current_layer_name)(current_n,2);
            TCmax_grat=max(TCu_observed.(current_layer_name)(current_n,:,1));
            TCmax_plaid=max(TCu_observed.(current_layer_name)(current_n,:,2));
            % initialize plot
            f2 = figure('units','normalized','outerposition',[0 0 1 1]);
            % set psth subplot position
            sb1=subplot(1,2,2);
            axis square
            % set psth sample rate
            sr=SR;
            % initialize handle array
            pipi=NaN(1,numel(stimulus_types));
            % loop over stimulus types
            for current_stimulus_type_id=1:numel(stimulus_types)
                % set color and tag to use
                if Clabel==2
                    coltuse=[255,150,0]./255;
                elseif Clabel==1
                    coltuse=[50,200,0]./255;
                elseif Clabel==0
                    coltuse=[150,150,150]./255;
                end
                hold on;
                % get psths
                psth_observed_y=PSTH_observed.(current_layer_name)(current_n,:,current_stimulus_type_id);
                psth_observed_x=(0:length(psth_observed_y)).*(1/sr);
                % plot psths
                pipi(current_stimulus_type_id)=plot(gca,psth_observed_x,[psth_observed_y(1),psth_observed_y],'-','Color',coltuse./(current_stimulus_type_id),'LineWidth',2.5);
                plot_shaded_auc(gca,psth_observed_x,[psth_observed_y(1),psth_observed_y],0.15,coltuse./(current_stimulus_type_id))
            end
            % embellish psth
            plot([0,0],[0,5],'--k', 'LineWidth',2)
            plot([1,1],[0,5],'--k', 'LineWidth',2)
            hlabelx=get(gca,'Xlabel');
            set(hlabelx,'String','time (s)','FontSize',12,'color','k')
            hlabely=get(gca,'Ylabel');
            set(hlabely,'String','normalized firing rate','FontSize',12,'color','k')
            legend(gca,pipi,{'grating','plaid'})
            % add extra info
            tt=text(0.05,3.2,['pDIR grating = ',num2str(pDIR_grat),' d'],'FontSize',12);
            ttt=text(0.05,3.1,['activity count grating = ',num2str(TCmax_grat)],'FontSize',12);
            tt2=text(0.05,2.6,['pDIR plaid = ',num2str(pDIR_plaid),' d'],'FontSize',12);
            ttt2=text(0.05,2.5,['activity count plaid = ',num2str(TCmax_plaid)],'FontSize',12);
            ylim([0,3.5])
            xlim([-0.2,1.2])
            title(['raster and psth (n=',num2str(current_n),') - DIR=',num2str(pDIR_grat),' - TF=','default',' - SF=','default'])
            set(gca,'FontSize',12);
            % set psth subplot position
            ppol=polaraxes('Position',[-0.05,0.19,.65,.65]);
            hold on;
            % loop over stimulus types
            for current_stimulus_type_id=1:numel(stimulus_types)
                % set color and tag to use
                if Clabel==2
                    coltuse=[255,150,0]./255;
                elseif Clabel==1
                    coltuse=[50,200,0]./255;
                elseif Clabel==0
                    coltuse=[150,150,150]./255;
                end
                TC=TCu_observed.(current_layer_name)(current_n,:,current_stimulus_type_id);
                obs_tc=[TC,TC(1)];
                % draw plar plots
                p2=polarplot(ppol,[deg2rad(DIR),2*pi],obs_tc,'-');
                set(p2,'color',coltuse./current_stimulus_type_id)
                set(p2, 'linewidth', 3.5);
            end
            % add info
            title(ppol,['polar plots (n=',num2str(current_n),' ',current_layer_name,') - DIR=',num2str(pDIR_grat),' - TF=','default',' - SF=','default'])
            tx2=text(ppol,deg2rad(45),ppol.RLim(end)*1.15,['obs DSI = ',num2str(DSI,'%.01f')],'fontsize',12);
            tx3=text(ppol,deg2rad(40),ppol.RLim(end)*1.15,['obs Zc = ',num2str(Zc,'%.01f')],'fontsize',12);
            tx5=text(ppol,deg2rad(30),ppol.RLim(end)*1.15,['obs Zp = ',num2str(Zp,'%.01f')],'fontsize',12);
            set(ppol,'fontsize',12);
            % save figure
            saveas(f2,[resultpath,filesep,'single_neu_example_n',num2str(current_n),'_',current_layer_name],'jpg')
            print(f2,'-depsc','-painters',[[resultpath,filesep,'single_neu_example_n',num2str(current_n),'_',current_layer_name],'.eps'])
            close all
        end
    end
    
end

%% save tuning analysis results -------------------------------------------------

% save relevant variables
save([resultpath,filesep,'Tuning_datastructure.mat'],...
    'layer_names',...
    'stimulus_types',...
    'PSTH_observed',...
    'TCu_observed',...
    'TC_observed',...
    'Zp_observed',...
    'Zc_observed',...
    'Clabel_observed',...
    'Rlabel_observed',...
    'DSI_observed',...
    'OSI_observed',...
    'pDIR_observed',...
    'Lid_observed');