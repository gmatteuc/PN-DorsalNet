% ------------------------- PREDICT PATTERN VERSUS COMPONENT WITH DORSALNET -------------------------

clear all %#ok<CLALL>
close all
clc
% NB: enitirely new for revision(26/06/2023)

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

% set wether to rerun regression analysis or reload results
bool_rerun_analysis=0;
% set whether to restrict to consistently classified neurons
bool_restrict_to_consitent=1;

% create prediction results folder
outputfolder=[resultpath,filesep,'representation_regression'];
if not(exist(outputfolder)) %#ok<EXIST>
    mkdir(outputfolder)
end

% load representations
DorsalNet_rdm=load([resultpath,filesep,'RDM_datastructure_DorsalNet.mat']);
Rat_rdm=load([resultpath,filesep,'RDM_datastructure_Rat.mat']); % NB: using the one in the local folder!
DorsalNet_repr=DorsalNet_rdm.obs_COUNT_per_class';
Rat_repr=Rat_rdm.obs_COUNT_per_class;
Rat_reprln=Rat_rdm.pred_COUNT_per_class;
% restrict to consistently classified neurons
if bool_restrict_to_consitent
    for i=1:numel(Rat_repr)
        valid_neu_idx=Rat_rdm.bullet_distr.idx_consist{i};
        Rat_repr{i}=Rat_repr{i}(valid_neu_idx,:);
        Rat_reprln{i}=Rat_reprln{i}(valid_neu_idx,:);
    end
end

%% run cross-validated population representation prediction ----------------

if or(bool_rerun_analysis,not(exist([resultpath,filesep,'representation_prediction_results.mat']))) %#ok<EXIST>
    
    % perform representation prediction ---
    
    % set labels for title
    pop_labels={'component','pattern','unclassified'};
    
    % initialize storage structures
    avg_cost_crossvals=cell([numel(pop_labels),numel(pop_labels)]);
    std_cost_crossvals=cell([numel(pop_labels),numel(pop_labels)]);
    predicted_mats_smooths=cell([numel(pop_labels),numel(pop_labels)]);
    
    % set fold n of cross-validation
    nfold = 24; % leave-one-out crossvalidation
    % set regularization parameter value
    regpar = 0.005;
    % set smoothing parameter value
    smoothpar = 0.5;
    
    %save resampling indeces
    uncl_resample_idx=cell(1,numel(pop_labels));
    
    % loop over populations
    for pop_idx1=1:numel(pop_labels)
        for pop_idx2=1:numel(pop_labels)
            
            % set input representations
            %             if pop_idx1==3
            %                 target_mat = Rat_repr{pop_idx1}(1:100,:);
            %             else
            target_mat = Rat_repr{pop_idx1};
            %             end
            if pop_idx2==3
                indexes=randsample(size(DorsalNet_repr{3},1),size(DorsalNet_repr{1},1));
                predictor_mat = DorsalNet_repr{pop_idx2}(indexes,:);
                uncl_resample_idx{pop_idx1}=indexes;
            else
                predictor_mat = DorsalNet_repr{pop_idx2};
            end
            % preprocess inputs (normalization, smoothing, reordering)
            [target_mat_smooth,~]=reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(target_mat, smoothpar, 1)));
            [predictor_mat_smooth,~]=reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(predictor_mat, smoothpar, 1)));
            % run cross-validated representation prediction
            tic
            %             [avg_cost_crossvals{pop_idx1,pop_idx2}, ...
            %                 std_cost_crossvals{pop_idx1,pop_idx2}, ...
            %                 predicted_mats_smooths{pop_idx1,pop_idx2}] = ...
            %                 get_crossvalidated_cost_predict_representation_parallel(...
            %                 target_mat_smooth, predictor_mat_smooth, regpar, nfold);
            [costs, predicted_mats] =...
                get_crossvalidated_cost_predict_representation_parallel(...
                target_mat_smooth, predictor_mat_smooth, regpar, nfold);
            avg_cost_crossvals{pop_idx1,pop_idx2} = nanmean(costs);
            std_cost_crossvals{pop_idx1,pop_idx2} = nanstd(costs);
            predicted_mats_smooths{pop_idx1,pop_idx2} = predicted_mats;
            toc
            % store title strings
            plottitlestrings{pop_idx1,pop_idx2}=['Rat ',pop_labels{pop_idx1},' population - DorsalNet ',pop_labels{pop_idx2},'-based prediction'];
            % output message
            disp( [pop_labels{pop_idx1},'-',pop_labels{pop_idx2},' cross-validated representation prediction completed ...'] )
            
        end
    end
    
    % save representation prediction results ---
    
    % save results
    save([resultpath,filesep,'representation_prediction_results.mat'],...
        'avg_cost_crossvals',...
        'std_cost_crossvals',...
        'predicted_mats_smooths',...
        'pop_labels',...
        'nfold',...
        'regpar',...
        'smoothpar');
    
else
    
    % load representation prediction results  ---
    
    % save results
    load([resultpath,filesep,'representation_prediction_results.mat']);
    
end

%% plot cross-validated population representation prediction ----------------

% get colors for plotting
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=[0,0,0]./(3*255);
% get labels for plotting
stimlabelstouse=Rat_rdm.labels;

% loop over populations
for pop_idx1=1:numel(pop_labels)
    
    % prepare figure
    fighand0=figure('units','normalized','outerposition',[0 0 1 1]);
    
    for pop_idx2=1:numel(pop_labels)
        
        % fetch data
        target_mat = Rat_repr{pop_idx1};
        targetln_mat = Rat_reprln{pop_idx1};
        %         if pop_idx1==3
        %             target_mat = Rat_repr{pop_idx1}(1:100,:);
        %             targetln_mat = Rat_reprln{pop_idx1}(1:100,:);
        %         else
        %             target_mat = Rat_repr{pop_idx1};
        %             targetln_mat = Rat_reprln{pop_idx1};
        %         end
        [target_mat_smooth,target_mat_reordperm] = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(target_mat, smoothpar, 1)));
        predicted_mats_smooth=predicted_mats_smooths{pop_idx1,pop_idx2};
        predicted_mat_smooth=squeeze(nanmean(predicted_mats_smooth,1));
        
        % compute new prediction accuracy
        errorvec=sqrt(nanmean((target_mat_smooth-predicted_mat_smooth).^2,2));
        neu_avg_error=nanmean(errorvec);
        neu_se_error=nanstd(errorvec)./sqrt(numel(errorvec));
        % store new prediction accuracy
        neu_avg_errors{pop_idx1,pop_idx2}=neu_avg_error; %#ok<*SAGROW>
        neu_se_errors{pop_idx1,pop_idx2}=neu_se_error;
        neu_errors{pop_idx1,pop_idx2}=errorvec;
        % get peak positions to display
        peak_positions_target = find_peak_positions(target_mat_smooth);
        peak_positions_predicted = find_peak_positions(predicted_mat_smooth);
        
        % set number of subplot rows
        n_subplot_row=3;
        
        % plot target representation matrix ---------
        subplot(3,n_subplot_row,1+n_subplot_row*(pop_idx2-1))
        imagesc(target_mat_smooth); colormap(gray); caxis([0,1]); colorbar; hold on;
        plot([12.5,12.5],[0.5,size(target_mat_smooth,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
        for j=1:size(peak_positions_target,1)
            scatter(peak_positions_target(j,1),[j],25,...
                'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                'markerfacealpha',1,'markeredgealpha',1)
            scatter(12+peak_positions_target(j,1),[j],25,...
                'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                'markerfacealpha',0.5,'markeredgealpha',0.5)
        end
        title('target representation')
        ylabel('neuron #')
        xticks([1:3:numel(stimlabelstouse)])
        xticklabels(stimlabelstouse(1:3:end));
        xtickangle(45)
        set(gca,'fontsize',12)
        axis square
        % plot predicted representation matrix ---------
        subplot(3,n_subplot_row,2+n_subplot_row*(pop_idx2-1))
        imagesc(predicted_mat_smooth); colormap(gray); caxis([0,1]); colorbar; hold on;
        plot([12.5,12.5],[0.5,size(target_mat_smooth,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
        for j=1:size(peak_positions_predicted,1)
            scatter(peak_positions_predicted(j,1),[j],25,...
                'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                'markerfacealpha',1,'markeredgealpha',1)
            scatter(12+peak_positions_predicted(j,1),[j],25,...
                'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                'markerfacealpha',0.5,'markeredgealpha',0.5)
        end
        title(['cross-val predicted representation (',pop_labels{pop_idx2},')'],'color',inputpars.distrcolors{pop_idx2})
        ylabel('neuron #')
        xticks([1:3:numel(stimlabelstouse)])
        xticklabels(stimlabelstouse(1:3:end));
        xtickangle(45)
        set(gca,'fontsize',12)
        axis square
        % plot predicted vs. observed difference matrix ---------
        subplot(3,n_subplot_row,3+n_subplot_row*(pop_idx2-1))
        imagesc(abs(target_mat_smooth-predicted_mat_smooth)); colormap(gray); colorbar; hold on;
        plot([12.5,12.5],[0.5,size(target_mat_smooth,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
        title(['target vs. predicted difference ( cost = ',num2str(round(neu_avg_error,3)),...
            ' +/- ',num2str(round(neu_se_error,3)),' )'],'color',inputpars.distrcolors{pop_idx2});
        ylabel('neuron #')
        xticks([1:3:numel(stimlabelstouse)])
        xticklabels(stimlabelstouse(1:3:end));
        xtickangle(45)
        set(gca,'fontsize',12)
        axis square
        
    end
    
    % add suptitle
    suptitle(['Rat ',pop_labels{pop_idx1},' population predictions'])
    % save
    saveas(fighand0,[resultpath,filesep,'Rat_',pop_labels{pop_idx1},'_population_prediction'],'jpg')
    print(fighand0,'-depsc','-painters',[[resultpath,filesep,'Rat_',pop_labels{pop_idx1},'_population_prediction'],'.eps'])
    
end

%% plot cross-validated population ln prediction ----------------

% get colors for plotting
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=[0,0,0]./(3*255);
% get labels for plotting
stimlabelstouse=Rat_rdm.labels;

% loop over populations
for pop_idx1=1:numel(pop_labels)
    
    % prepare figure
    fighand0=figure('units','normalized','outerposition',[0 0 1 1]);
    
    % fetch data
    target_mat = Rat_repr{pop_idx1};
    targetln_mat = Rat_reprln{pop_idx1};
    %     if pop_idx1==3
    %         target_mat = Rat_repr{pop_idx1}(1:100,:);
    %         targetln_mat = Rat_reprln{pop_idx1}(1:100,:);
    %     else
    %         target_mat = Rat_repr{pop_idx1};
    %         targetln_mat = Rat_reprln{pop_idx1};
    %     end
    targetln_mat(targetln_mat<=0)=0;
    [target_mat_smooth,target_mat_reordperm] = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(target_mat, smoothpar, 1)));
    temp = max_normalize_halves(gaussianSmooth1D(targetln_mat, smoothpar, 1));
    targetln_mat_smooth = temp(target_mat_reordperm, :);
    
    % compute new prediction accuracy
    errorvec=sqrt(nanmean((target_mat_smooth-targetln_mat_smooth).^2,2));
    neu_avg_error=nanmean(errorvec);
    neu_se_error=nanstd(errorvec)./sqrt(numel(errorvec));
    % store new prediction accuracy
    neu_avg_errors_ln{pop_idx1}=neu_avg_error; %#ok<*SAGROW>
    neu_se_errors_ln{pop_idx1}=neu_se_error;
    neu_errors_ln{pop_idx1}=errorvec; %#ok<*SAGROW>
    
    % get peak positions to display
    peak_positions_target = find_peak_positions(target_mat_smooth);
    peak_positions_targetln = find_peak_positions(targetln_mat_smooth);
    
    % set number of subplot rows
    n_subplot_row=3;
    
    % plot target representation matrix ---------
    subplot(1,n_subplot_row,1)
    imagesc(target_mat_smooth); colormap(gray); caxis([0,1]); colorbar; hold on;
    plot([12.5,12.5],[0.5,size(target_mat_smooth,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
    for j=1:size(peak_positions_target,1)
        scatter(peak_positions_target(j,1),[j],25,...
            'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
            'markerfacealpha',1,'markeredgealpha',1)
        scatter(12+peak_positions_target(j,1),[j],25,...
            'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
            'markerfacealpha',0.5,'markeredgealpha',0.5)
    end
    title('target representation')
    ylabel('neuron #')
    xticks([1:numel(stimlabelstouse)])
    xticklabels(stimlabelstouse);
    xtickangle(45)
    axis square
    set(gca,'fontsize',12)
    % plot ln predicted representation matrix ---------
    subplot(1,n_subplot_row,2)
    imagesc(targetln_mat_smooth); colormap(gray); caxis([0,1]); colorbar; hold on;
    plot([12.5,12.5],[0.5,size(targetln_mat_smooth,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
    for j=1:size(peak_positions_targetln,1)
        scatter(peak_positions_targetln(j,1),[j],25,...
            'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
            'markerfacealpha',1,'markeredgealpha',1)
        scatter(12+peak_positions_targetln(j,1),[j],25,...
            'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
            'markerfacealpha',0.5,'markeredgealpha',0.5)
    end
    title(['cross-val ln-predicted representation (','ln',')'],'color',inputpars.distrcolors{1}.*0.33)
    ylabel('neuron #')
    xticks([1:numel(stimlabelstouse)])
    xticklabels(stimlabelstouse);
    xtickangle(45)
    axis square
    set(gca,'fontsize',12)
    % plot predicted vs. observed difference matrix ---------
    subplot(1,n_subplot_row,3)
    imagesc(abs(target_mat_smooth-targetln_mat_smooth)); colormap(gray); colorbar; hold on;
    plot([12.5,12.5],[0.5,size(targetln_mat_smooth,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
    title(['target vs. predicted difference ( cost = ',num2str(round(neu_avg_error,3)),...
        ' +/- ',num2str(round(neu_se_error,3)),' )'],'color',inputpars.distrcolors{1}.*0.33);
    ylabel('neuron #')
    xticks([1:numel(stimlabelstouse)])
    xticklabels(stimlabelstouse);
    xtickangle(45)
    axis square
    set(gca,'fontsize',12)
    
    % add suptitle
    suptitle(['Rat ',pop_labels{pop_idx1},' population predictions'])
    % save
    saveas(fighand0,[resultpath,filesep,'Rat_',pop_labels{pop_idx1},'_ln_prediction'],'jpg')
    print(fighand0,'-depsc','-painters',[[resultpath,filesep,'Rat_',pop_labels{pop_idx1},'_ln_prediction'],'.eps'])
    
end

%% plot cross-validated population representation prediction comparison ----------------

% convert cell arrays to matrices
avg_cost_crossvals_to_use = cell2mat(neu_avg_errors)';
std_cost_crossvals_to_use = cell2mat(neu_se_errors)';
avg_cost_crossvals_ln_to_use = cell2mat(neu_avg_errors_ln)';
std_cost_crossvals_ln_to_use = cell2mat(neu_se_errors_ln)';
avg_cost_crossvals_to_use = avg_cost_crossvals_to_use(:);
std_cost_crossvals_to_use = std_cost_crossvals_to_use(:);
% perform statistics to compare prediction quality
errdistrlabels={...
    'comp-comp','comp-patt','comp-uncl','comp-ln',...
    'patt-comp','patt-patt','patt-uncl','patt-ln',...
    'uncl-comp','uncl-patt','uncl-uncl','uncl-ln'};
errdistrcolors={...
    inputpars.distrcolors{1},inputpars.distrcolors{1},inputpars.distrcolors{1},inputpars.distrcolors{1}...
    inputpars.distrcolors{2},inputpars.distrcolors{2},inputpars.distrcolors{2},inputpars.distrcolors{2}...
    inputpars.distrcolors{3},inputpars.distrcolors{3},inputpars.distrcolors{3},inputpars.distrcolors{3}};
errdistr=[neu_errors(1,:),neu_errors_ln(1),neu_errors(2,:),neu_errors_ln(2),neu_errors(3,:),neu_errors_ln(3)];
errcompmatp=NaN(numel(errdistr),numel(errdistr));
errcompmats=NaN(numel(errdistr),numel(errdistr));
for ii=1:numel(errdistr)
    for jj=1:numel(errdistr)
        % perform ttest between error distributions
        [currh,currp]=ttest2(errdistr{ii},errdistr{jj});
        % store ttest p values
        errcompmatp(ii,jj)=currp;
        % store ttest surprise values
        errcompmats(ii,jj)=-log10(currp);
    end
end

% prepare figure
fighand1=figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,3,[1,2])
hold on;
% comp ----
b1 = bar([1],avg_cost_crossvals_to_use(1),...
    'facecolor',inputpars.distrcolors{1},...
    'edgecolor',inputpars.distrcolors{1}); %#ok<NASGU>
b2 = bar([2],avg_cost_crossvals_to_use(2),...
    'facecolor',inputpars.distrcolors{1}.*0.75,...
    'edgecolor',inputpars.distrcolors{1}.*0.75); %#ok<NASGU>
b3 = bar([3],avg_cost_crossvals_to_use(3),...
    'facecolor',(inputpars.distrcolors{1}.*0.33+[1,1,1])./2,...
    'edgecolor',(inputpars.distrcolors{1}.*0.33+[1,1,1])./2); %#ok<NASGU>
b4 = bar([4],avg_cost_crossvals_ln_to_use(1),...
    'facecolor',inputpars.distrcolors{1}.*0.33,...
    'edgecolor',inputpars.distrcolors{1}.*0.33);
% patt ----
b5 = bar([5],avg_cost_crossvals_to_use(4),...
    'facecolor',inputpars.distrcolors{2}.*0.75,...
    'edgecolor',inputpars.distrcolors{2}.*0.75);
b6 = bar([6],avg_cost_crossvals_to_use(5),...
    'facecolor',inputpars.distrcolors{2},...
    'edgecolor',inputpars.distrcolors{2});
b7 = bar([7],avg_cost_crossvals_to_use(6),...
    'facecolor',(inputpars.distrcolors{2}.*0.33+[1,1,1])./2,...
    'edgecolor',(inputpars.distrcolors{2}.*0.33+[1,1,1])./2);
b8 = bar([8],avg_cost_crossvals_ln_to_use(2),...
    'facecolor',inputpars.distrcolors{2}.*0.33,...
    'edgecolor',inputpars.distrcolors{2}.*0.33);
% uncl ----
b9 = bar([9],avg_cost_crossvals_to_use(7),...
    'facecolor',inputpars.distrcolors{3}.*0.75,...
    'edgecolor',inputpars.distrcolors{3}.*0.75);
b10 = bar([10],avg_cost_crossvals_to_use(8),...
    'facecolor',inputpars.distrcolors{3},...
    'edgecolor',inputpars.distrcolors{3});
b11 = bar([11],avg_cost_crossvals_to_use(9),...
    'facecolor',(inputpars.distrcolors{3}.*0.33+[1,1,1])./2,...
    'edgecolor',(inputpars.distrcolors{3}.*0.33+[1,1,1])./2);
b12 = bar([12],avg_cost_crossvals_ln_to_use(3),...
    'facecolor',inputpars.distrcolors{3}.*0.33,...
    'edgecolor',inputpars.distrcolors{3}.*0.33);

hold on;
% add error bars - non ln
errorbar([1,2,3,5,6,7,9,10,11], avg_cost_crossvals_to_use, std_cost_crossvals_to_use,...
    'k', 'linewidth', 2, 'linestyle', 'none');
scatter([1,2,3,5,6,7,9,10,11], avg_cost_crossvals_to_use,75,'Markerfacecolor',[0,0,0],'Markeredgecolor',[0,0,0]);
% add error bars - ln
errorbar([4,8,12], avg_cost_crossvals_ln_to_use, std_cost_crossvals_ln_to_use,...
    'k', 'linewidth', 2, 'linestyle', 'none');
scatter([4,8,12], avg_cost_crossvals_ln_to_use,75,'Markerfacecolor',[0,0,0],'Markeredgecolor',[0,0,0]);
plot([4.5,4.5],[0.15,0.45],':','linewidth',2,'color',[0.5,0.5,0.5])
plot([8.5,8.5],[0.15,0.45],':','linewidth',2,'color',[0.5,0.5,0.5])
% add labels and title
% xticks([1,2,3,4]);
% xlim([0,7])
xlim([0,13])
ylim([0.15,0.45])
set(gca,'fontsize',12)
% xticks([1:6]);
xticks([1:12]);
xtickangle(45)
xticklabels(errdistrlabels);
ylabel('avg rmse (cross-validated)');
axis square
title('representation prediction rmses');
subplot(1,3,3)
imagesc(errcompmats); colorbar; colormap(gray); caxis([0,10]);
xticks(1:numel(errdistrlabels))
yticks(1:numel(errdistrlabels))
xticklabels(errdistrlabels)
yticklabels(errdistrlabels)
xtickangle(45)
axis square
ax = gca;
cm=cell2mat(errdistrcolors');
for itick = 1:numel(ax.YTickLabel)
    ax.YTickLabel{itick} = ...
        sprintf('\\color[rgb]{%f,%f,%f}%s',...
        cm(itick,:),...
        ax.YTickLabel{itick});
    ax.XTickLabel{itick} = ...
        sprintf('\\color[rgb]{%f,%f,%f}%s',...
        cm(itick,:),...
        ax.XTickLabel{itick});
end
astkthreshold = -log10(0.05);
[nrows, ncols] = size(errcompmats);
hold on;
for r = 1:nrows
    for c = 1:ncols
        if errcompmats(r,c) > astkthreshold
            if r<=3 && c<=3
                colortouse=inputpars.distrcolors{1};
            elseif r>=3 && c>=3
                colortouse=inputpars.distrcolors{2};
            else
                colortouse=(inputpars.distrcolors{1}+inputpars.distrcolors{2})/2;
            end
            plot(c, r, '*', 'MarkerSize', 15, 'Color',colortouse);
        end
    end
end
set(gca,'fontsize',12)
title('avg rmse comparison (ttest) - surp matrix')
sgtitle('Rat representation prediction comparison (DorsalNet populations and LN)')

% save
saveas(fighand1,[resultpath,filesep,'Rat_population_prediction_comparison'],'jpg')
print(fighand1,'-depsc','-painters',[['Rat_population_prediction_comparison'],'.eps'])

%% reclassify in quasi-pattern and quasi-component based on reconstruction error

% initialize datastructures
neu_errormats=cell(1,numel(pop_labels));
quasi_classlbls=cell(1,numel(pop_labels));
quasi_classlbls_conf=cell(1,numel(pop_labels));
true_classlbls=cell(1,numel(pop_labels));
true_quasi_agreement=NaN(1,numel(pop_labels));
% collect neuron errormats
for pop_idx1=1:numel(pop_labels)
    neu_errormats{pop_idx1}=[neu_errors{pop_idx1,1},neu_errors{pop_idx1,2},neu_errors{pop_idx1,3}];
end
% find minimum in neuron errormats to perform quasi-classification
for pop_idx1=1:numel(pop_labels)
    % get quasi class labels
    [quasi_classlbls_minval,quasi_classlbls{pop_idx1}]=min(neu_errormats{pop_idx1},[],2);
    tempbasl=(nansum(neu_errormats{pop_idx1},2)-quasi_classlbls_minval)./2;
    quasi_classlbls_conf{pop_idx1}=abs(quasi_classlbls_minval-tempbasl)./tempbasl;
    % get true class labels
    true_classlbls{pop_idx1}=pop_idx1.*ones(size(neu_errormats{pop_idx1},1),1);
    % compute agreement
    true_quasi_agreement(pop_idx1)=nansum(true_classlbls{pop_idx1}==quasi_classlbls{pop_idx1})./sum(not(isnan(true_classlbls{pop_idx1})));
end

%% plot cross-validated population representation quasi-classification analysis ----------------

% initialize bullet comparison variables
Zc_obs=cell(3,1);
Zp_obs=cell(3,1);
Zc_pred_ln=cell(3,1);
Zp_pred_ln=cell(3,1);
Zc_pred_rpr=cell(3,2);
Zp_pred_rpr=cell(3,2);
ctlabel_pred_rpr=cell(3,1);
ctlabel_pred_ln=cell(3,1);
ctlabel_quasi=cell(3,1);
ctlabel_quasi_conf=cell(3,1);
% organize data to get bullet plots
for pop_idx1=1:3
    % get indeces of valid neurons
    %     if pop_idx1==3
    %         idx_valid_neu=Rat_rdm.bullet_distr.idx_consist{pop_idx1}(1:100);
    %     else
    idx_valid_neu=Rat_rdm.bullet_distr.idx_consist{pop_idx1};
    %     end
    % partial correlations
    Zc_obs{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zc_O_distribs{pop_idx1}(idx_valid_neu);
    Zp_obs{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zp_O_distribs{pop_idx1}(idx_valid_neu);
    Zc_pred_ln{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zc_P_distribs{pop_idx1}(idx_valid_neu);
    Zp_pred_ln{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zp_P_distribs{pop_idx1}(idx_valid_neu);
    % loop over types of representation prediction
    for pop_idx2=1:2
        current_pred_rprs=predicted_mats_smooths{pop_idx1,pop_idx2};
        current_pred_rpr=squeeze(nanmean(current_pred_rprs,1));
        % initialize Zp and Zc storage
        Zc_pred_rpr{pop_idx1,pop_idx2}=NaN(size(current_pred_rpr,1),1);
        Zp_pred_rpr{pop_idx1,pop_idx2}=NaN(size(current_pred_rpr,1),1);
        % initializeclassification storage
        ctlabel_pred_rpr{pop_idx1}=NaN(size(current_pred_rpr,1),1);
        ctlabel_pred_ln{pop_idx1}=NaN(size(current_pred_rpr,1),1);
        for neu_idx=1:size(current_pred_rpr,1)
            % perform partial correlation analysis
            curr_grat_tc=current_pred_rpr(neu_idx,1:12);
            curr_plaid_tc=current_pred_rpr(neu_idx,12+(1:12));
            [ ~, ~, temp_Zp, temp_Zc, ~, ~, ~, ~, ~, ~ ] =...
                get_pattern_index( curr_grat_tc,curr_plaid_tc );
            % store results
            Zc_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=temp_Zc;
            Zp_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=temp_Zp;
            % store predicted classifications - rpr
            if temp_Zp-max(temp_Zc,0)>=1.28
                ctlabel_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=2; % 2=pattern
            elseif temp_Zc-max(temp_Zp,0)>=1.28
                ctlabel_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=1; % 1=component
            else
                ctlabel_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=0; % 0=unclassified
            end
            % store predicted classifications - ln
            if Zp_pred_ln{pop_idx1}(neu_idx)-max(Zc_pred_ln{pop_idx1}(neu_idx),0)>=1.28
                ctlabel_pred_ln{pop_idx1}(neu_idx)=2; % 2=pattern
            elseif Zc_pred_ln{pop_idx1}(neu_idx)-max(Zp_pred_ln{pop_idx1}(neu_idx),0)>=1.28
                ctlabel_pred_ln{pop_idx1}(neu_idx)=1; % 1=component
            else
                ctlabel_pred_ln{pop_idx1}(neu_idx)=0; % 0=unclassified
            end
        end
        % collect quasi labels
        ctlabel_quasi{pop_idx1}=quasi_classlbls{pop_idx1};
        ctlabel_quasi_conf{pop_idx1}=quasi_classlbls_conf{pop_idx1};
    end
end

% initialize figure
fighand1bis = figure('units','normalized','outerposition',[0 0 1 1]);
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=[0,0,0]./(3*255);
confthreshold=0.30;
% plot bullet plot observed with classification overlay ----------------------------
subplot(2,3,[1,2,4,5])
hold on;
transparentconfidencebool=0;
% loop over neurons
for pop_idx1=1:3
    for current_n=1:numel(Zc_obs{pop_idx1})
        % get data to plot
        Zc_O=Zc_obs{pop_idx1}(current_n);
        Zp_O=Zp_obs{pop_idx1}(current_n);
        % get quasicolor
        current_quasilabel=ctlabel_quasi{pop_idx1}(current_n);
        current_quasiconf=ctlabel_quasi_conf{pop_idx1}(current_n);
        quasicolor=inputpars.distrcolors{current_quasilabel};
        % plot in bullet
        scatter(Zc_O,Zp_O,150,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        if transparentconfidencebool
            scatter(Zc_O,Zp_O,30,...
                'MarkerFaceColor',[1,1,1],'MarkerEdgeColor',[1,1,1],...
                'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1); %#ok<UNRCH>
            scatter(Zc_O,Zp_O,30,...
                'MarkerFaceColor',quasicolor,'MarkerEdgeColor',quasicolor,...
                'MarkerFaceAlpha',min([current_quasiconf./confthreshold,1]),'MarkerEdgeAlpha',min([current_quasiconf./confthreshold,1]));
        else
            scatter(Zc_O,Zp_O,30,...
                'MarkerFaceColor',quasicolor,'MarkerEdgeColor',quasicolor,...
                'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        end
    end
end
line([0 9], [1.28 10.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
line([1.28 10.28], [0 9],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
line([1.28 1.28], [-5 0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
line([-5 0], [1.28 1.28],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
xlim([-4,6.5])
ylim([-4,6.5])
axis square
xlabel('Zc'); ylabel('Zp');
title(['Zp vs. Zc ',' classification vs- quasi-classification']);
set(gca,'fontsize',12)
% barplot of consistency -----------------------------------------------------
subplot(2,3,6)
hold on;
% comp ----
b1 = bar([1],true_quasi_agreement(1),...
    'facecolor',inputpars.distrcolors{1},...
    'edgecolor',inputpars.distrcolors{1});
b2 = bar([2],true_quasi_agreement(2),...
    'facecolor',inputpars.distrcolors{2},...
    'edgecolor',inputpars.distrcolors{2});
b3 = bar([3],true_quasi_agreement(3),...
    'facecolor',(inputpars.distrcolors{3}),...
    'edgecolor',(inputpars.distrcolors{3}));
xlim([0,4])
ylim([0,1])
set(gca,'fontsize',12)
xticks([1:3]);
xtickangle(45)
xticklabels({'component','pattern','unclassified'});
ylabel('fraction consistent');
title('consistency between classifications')
% violin plot of confidence -----------------------------------------------------
subplot(2,3,3)
hold on;
ylabellist={'component','pattern','unclassified'};
% decide wheter to use max or unrolled
distribtouse=quasi_classlbls_conf; % collapsed_max_fitted_rfs_r2_distribs
inputpars.inputaxh=gca;
hold(inputpars.inputaxh,'on')
% set settings for violin distribution plotting
inputpars.boxplotwidth=0.4;%0.5;
inputpars.boxplotlinewidth=2;
inputpars.densityplotwidth=0.4;%0.5;
inputpars.yimtouse=[0,0.7];
% inputpars.yimtouse=[0,8];
inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
inputpars.scatteralpha=0.15;
inputpars.scattersize=40;
inputpars.distralpha=0.5;
inputpars.xlabelstring=[];
inputpars.ylabelstring='confidence score';
inputpars.titlestring=['quasi-classification confidence'];
inputpars.boolscatteron=1;
inputpars.ks_bandwidth=0.05;
inputpars.xlimtouse=[-0.5,4.5];
% plot violins
inputadata.inputdistrs=distribtouse;
inputpars.n_distribs=numel(inputadata.inputdistrs);
inputpars.dirstrcenters=(1:inputpars.n_distribs);
inputpars.xtickslabelvector={'comp','patt','uncl'};
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=[0,0,0]./355;
[~,scatter_xs,scatter_ys] = plot_violinplot_PN_new(inputadata,inputpars);
[~,pval_comp_patt] = ttest2(distribtouse{1},distribtouse{2});
[~,pval_comp_uncl] = ttest2(distribtouse{1},distribtouse{3});
[~,pval_patt_uncl] = ttest2(distribtouse{2},distribtouse{3});
text(-0.25,0.68,['comp-patt mean diff p = ',num2str(pval_comp_patt)],'fontsize',11)
text(-0.25,0.65,['comp-uncl mean diff p = ',num2str(pval_comp_uncl)],'fontsize',11)
text(-0.25,0.62,['patt-uncl mean diff p = ',num2str(pval_patt_uncl)],'fontsize',11)
xtickangle(45)
set(gca,'fontsize',12)
axis square
hold on;
line([-0.5,5.5],[confthreshold,confthreshold],'LineStyle','--','LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% add suptitle
sgtitle('representation prediction-based "quasi" classification')
% save
saveas(fighand1bis,[resultpath,filesep,'quasi_classification_diagnostics'],'jpg')
print(fighand1bis,'-depsc','-painters',[resultpath,filesep,['quasi_classification_diagnostics'],'.eps'])

%% plot qPI vs. PI scatter ----------------

% initialize figure
fighand1tris = figure('units','normalized','outerposition',[0 0 1 1]);
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
inputpars.distrcolors{3}=[0,0,0]./(3*255);
all_PI_O=[];
all_qPI_O=[];
% loop over neurons
for pop_idx1=1:3
    hold on;
    % get data to plot
    Zc_O=Zc_obs{pop_idx1};
    Zp_O=Zp_obs{pop_idx1};
    PI_O=Zp_O-Zc_O;
    current_quasilabel=ctlabel_quasi{pop_idx1};
    current_quasiconf=ctlabel_quasi_conf{pop_idx1};
    current_quasilabel(current_quasilabel==1)=-1;
    current_quasilabel(current_quasilabel==2)=1;
    current_quasilabel(current_quasilabel==3)=0;
    qPI_O=current_quasiconf.*current_quasilabel;
    % plot in bullet
    scatter(PI_O,qPI_O,150,...
        'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
        'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
    % store plotted data
    all_PI_O=cat(1,all_PI_O,PI_O);
    all_qPI_O=cat(1,all_qPI_O,qPI_O);
end
xtofit=all_PI_O;
ytofit=all_qPI_O;
xpredlims=[-7,7];
ypredlims=[-0.8,+0.8];
[ypred,ypred_ci,xpred,opt_b,opt_b_ci,lmodel] = get_lin_fit_with_ci(xtofit,ytofit,xpredlims);
opt_b_halfciwidth=nanmedian(abs(opt_b_ci(2,:)-opt_b(2)));
[v_rho_PI_qPI,p_rho_PI_qPI] = corr(xtofit,ytofit);
colll=[0.5,0.5,0.5];
[hmu,hstd] = plot_shaded_mu_std(gca,xpred,ypred,ypred_ci,colll,0.15);
plot(gca,xpred,ypred,'-','linewidth',2,'color',colll)
plot(gca,[0,0],ypredlims,':','linewidth',2,'color',colll)
plot(gca,xpredlims,[0,0],':','linewidth',2,'color',colll)
text(-6,0.75,['corr v = ',num2str(round(v_rho_PI_qPI,3))],'fontsize',12)
text(-6,0.71,['corr p = ',num2str(p_rho_PI_qPI)],'fontsize',12)
text(-6,0.67,['slope = ',num2str(round(opt_b(2),3)),' +/- ',num2str(round(opt_b_halfciwidth,3))],'fontsize',12)
xlim(xpredlims)
ylim(ypredlims)
ylabel('PI index')
xlabel('quasi-PI score')
title(['PI vs. quasi-PI scatter (n=',num2str(numel(all_qPI_O)),')'])
set(gca,'fontsize',12)
% save
saveas(fighand1tris,[resultpath,filesep,'quasi_classification_diagnostics_bis'],'jpg')
print(fighand1tris,'-depsc','-painters',[resultpath,filesep,['quasi_classification_diagnostics_bis'],'.eps'])

%% compare patt. and comp. with quasi-patt. and quasi-comp. grand averages ----------------

quasiconf_to_use=ctlabel_quasi_conf{3};
quasilabels_to_use=ctlabel_quasi{3};
confthreshold=0.3;
valid_neus=find(quasiconf_to_use>=confthreshold);
% segregate unclassified by quasilabel
Rat_repr_quasi{1}=Rat_repr{3}(ctlabel_quasi{3}(valid_neus)==1,:);
Rat_repr_quasi{2}=Rat_repr{3}(ctlabel_quasi{3}(valid_neus)==2,:);
Rat_repr_quasi{3}=Rat_repr{3}(ctlabel_quasi{3}(valid_neus)==3,:);
% update labels
tempdirs=0:30:330;
tempdirs_bis=(0:30:330)-30*5;
stimlabelstousebis=cell(1,numel(stimlabelstouse));
for k=1:numel(stimlabelstouse)
    idxxx=mod(k,12);
    idxxx(idxxx==0)=12;
    stimlabelstousebis{k}=strrep(stimlabelstouse{k},num2str(tempdirs(idxxx)),num2str(tempdirs_bis(idxxx)));
end
% set whether to realign tc to grating peak or not
boolrecenteredmatrix=0;

fighand1quater = figure('units','normalized','outerposition',[0 0 1 1]);
for pop_idx1=1:2%numel(pop_labels)
    % plot non- quasi repr avergaes
    input_repr=Rat_repr{pop_idx1};
    preprocessed_repr=reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
    rearranged_responses = recenter_neuron_responses(preprocessed_repr, 6);
    [rearranged_responses_bis,rearranged_responses_bis_reordperm] = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
    peak_positions_rearranged_responses_bis = find_peak_positions(rearranged_responses_bis);
    rearranged_responses_avg = nanmean(rearranged_responses,1);
    rearranged_responses_se = nanstd(rearranged_responses,1)./sqrt(size(rearranged_responses,1));
    subplot(2,4,1+4*(pop_idx1-1))
    if boolrecenteredmatrix
        imagesc(rearranged_responses); %#ok<UNRCH>
        colormap(gray); caxis([0,1]);
        hold on;
        plot([6,6],[0.5,size(rearranged_responses,1)+0.5],'-','color',[inputpars.distrcolors{pop_idx1},0.5],'linewidth',2)
        plot([12+6,12+6],[0.5,size(rearranged_responses,1)+0.5],'-','color',[inputpars.distrcolors{pop_idx1},0.5],'linewidth',2)
    else
        imagesc(rearranged_responses_bis);
        colormap(gray); caxis([0,1]);
        hold on;
        for j=1:size(peak_positions_rearranged_responses_bis,1)
            scatter(peak_positions_rearranged_responses_bis(j,1),[j],25,...
                'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                'markerfacealpha',1,'markeredgealpha',1)
            scatter(12+peak_positions_rearranged_responses_bis(j,1),[j],25,...
                'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                'markerfacealpha',0.5,'markeredgealpha',0.5)
        end
    end
    plot([12.5,12.5],[0.5,size(rearranged_responses,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
    title([pop_labels{pop_idx1},' reord. repr-mat'],'color',inputpars.distrcolors{pop_idx1});
    ylabel('neuron #')
    xticks([1:3:numel(stimlabelstousebis)])
    xticklabels(stimlabelstousebis(1:3:end));
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
    subplot(2,4,2+4*(pop_idx1-1))
    hold on;
    grat_tc_avg=rearranged_responses_avg(1:12);
    grat_tc_se=rearranged_responses_se(1:12);
    plaid_tc_avg=rearranged_responses_avg(13:24);
    plaid_tc_se=rearranged_responses_se(13:24);
    [ ~, ~, Zp, Zc, ~, ~, ~, ~, ~, ~ ] = ...
        get_pattern_index( grat_tc_avg',plaid_tc_avg' );
    alphatouse=0.15;
    xvals=(0:30:330)-30*5;
    colortouse=inputpars.distrcolors{pop_idx1};
    plot_shaded_mu_std(gca,xvals,grat_tc_avg,grat_tc_se,colortouse,alphatouse);
    plot(xvals,grat_tc_avg,'-','color',colortouse,'linewidth',2)
    colortouse=inputpars.distrcolors{pop_idx1}.*0.75;
    plot_shaded_mu_std(gca,xvals,plaid_tc_avg,plaid_tc_se,colortouse,alphatouse);
    text(-140,1,['Zc = ',num2str(round(Zc,2))],'fontsize',12)
    text(-140,0.95,['Zp = ',num2str(round(Zp,2))],'fontsize',12)
    text(-140,0.90,['PI = ',num2str(round(Zp-Zc,2))],'fontsize',12)
    plot(xvals,plaid_tc_avg,'-','color',colortouse,'linewidth',2)
    title([pop_labels{pop_idx1},' avg. tc'],'color',inputpars.distrcolors{pop_idx1});
    ylabel('normalized response')
    xlabel('direction')
    ylim([0.15,1.1])
    xlim([min(xvals),max(xvals)]);
    set(gca,'fontsize',12)
    plot([0,0],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
    plot([60,60],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
    plot([-60,-60],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
    if pop_idx1~=3
        % plot quasi repr avergaes
        input_repr=Rat_repr_quasi{pop_idx1};
        preprocessed_repr=reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
        rearranged_responses = recenter_neuron_responses(preprocessed_repr, 6);
        [rearranged_responses_bis,rearranged_responses_bis_reordperm] = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
        peak_positions_rearranged_responses_bis = find_peak_positions(rearranged_responses_bis);
        rearranged_responses_avg = nanmean(rearranged_responses,1);
        rearranged_responses_se = nanstd(rearranged_responses,1)./sqrt(size(rearranged_responses,1));
        subplot(2,4,3+4*(pop_idx1-1))
        if boolrecenteredmatrix
            imagesc(rearranged_responses); %#ok<UNRCH>
            colormap(gray); caxis([0,1]);
            hold on;
            plot([6,6],[0.5,size(rearranged_responses,1)+0.5],'-','color',[inputpars.distrcolors{pop_idx1},0.5],'linewidth',2)
            plot([12+6,12+6],[0.5,size(rearranged_responses,1)+0.5],'-','color',[inputpars.distrcolors{pop_idx1},0.5],'linewidth',2)
        else
            imagesc(rearranged_responses_bis);
            colormap(gray); caxis([0,1]);
            hold on;
            for j=1:size(peak_positions_rearranged_responses_bis,1)
                scatter(peak_positions_rearranged_responses_bis(j,1),[j],25,...
                    'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                    'markerfacealpha',1,'markeredgealpha',1)
                scatter(12+peak_positions_rearranged_responses_bis(j,1),[j],25,...
                    'markerfacecolor',inputpars.distrcolors{pop_idx1},'markeredgecolor',inputpars.distrcolors{pop_idx1},...
                    'markerfacealpha',0.5,'markeredgealpha',0.5)
            end
            
        end
        plot([12.5,12.5],[0.5,size(rearranged_responses,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
        title(['quasi-',pop_labels{pop_idx1},' reord. repr-mat'],'color',inputpars.distrcolors{pop_idx1});
        ylabel('neuron #')
        xticks([1:3:numel(stimlabelstousebis)])
        xticklabels(stimlabelstousebis(1:3:end));
        xtickangle(45)
        set(gca,'fontsize',12)
        axis square
        subplot(2,4,4+4*(pop_idx1-1))
        hold on;
        grat_tc_avg=rearranged_responses_avg(1:12);
        grat_tc_se=rearranged_responses_se(1:12);
        plaid_tc_avg=rearranged_responses_avg(13:24);
        plaid_tc_se=rearranged_responses_se(13:24);
        [ ~, ~, Zp, Zc, ~, ~, ~, ~, ~, ~ ] = ...
            get_pattern_index( grat_tc_avg',plaid_tc_avg' );
        alphatouse=0.15;
        xvals=(0:30:330)-30*5;
        colortouse=inputpars.distrcolors{pop_idx1};
        plot_shaded_mu_std(gca,xvals,grat_tc_avg,grat_tc_se,colortouse,alphatouse);
        plot(xvals,grat_tc_avg,'-','color',colortouse,'linewidth',2)
        colortouse=inputpars.distrcolors{pop_idx1}.*0.75;
        plot_shaded_mu_std(gca,xvals,plaid_tc_avg,plaid_tc_se,colortouse,alphatouse);
        text(-140,1,['Zc = ',num2str(round(Zc,2))],'fontsize',12)
        text(-140,0.95,['Zp = ',num2str(round(Zp,2))],'fontsize',12)
        text(-140,0.90,['PI = ',num2str(round(Zp-Zc,2))],'fontsize',12)
        plot(xvals,plaid_tc_avg,'-','color',colortouse,'linewidth',2)
        title(['quasi-',pop_labels{pop_idx1},' avg. tc'],'color',inputpars.distrcolors{pop_idx1});
        ylabel('normalized response')
        xlabel('direction')
        ylim([0.15,1.1])
        xlim([min(xvals),max(xvals)]);
        set(gca,'fontsize',12)
        plot([0,0],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
        plot([60,60],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
        plot([-60,-60],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
    end
end
sgtitle('comparison of patt. and comp. with quasi-patt. and quasi-comp.')
% save
saveas(fighand1quater,[resultpath,filesep,'quasi_classification_diagnostics_tris'],'jpg')
print(fighand1quater,'-depsc','-painters',[resultpath,filesep,['quasi_classification_diagnostics_tris'],'.eps'])
close all

% % find max plaid dir on realigned matrices ----
% dirs=0:30:330;
% dirdiff=dirs-5*30;
% input_repr=Rat_repr{1};
% plaid_responses_temp=reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
% plaid_responses_temp = recenter_neuron_responses(plaid_responses_temp, 6);
% plaid_responses_temp = plaid_responses_temp(:, 13:24);
% [~, max_indices_comp] = max(plaid_responses_temp, [], 2);
% inputdatatemp{1}=abs(dirdiff(max_indices_comp));
% input_repr=Rat_repr{2};
% plaid_responses_temp = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
% plaid_responses_temp = recenter_neuron_responses(plaid_responses_temp, 6);
% plaid_responses_temp = plaid_responses_temp(:, 13:24);
% [~, max_indices_patt] = max(plaid_responses_temp, [], 2);
% inputdatatemp{2}=abs(dirdiff(max_indices_patt));
% input_repr=Rat_repr_quasi{1};
% plaid_responses_temp = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
% plaid_responses_temp = recenter_neuron_responses(plaid_responses_temp, 6);
% plaid_responses_temp = plaid_responses_temp(:, 13:24);
% [~, max_indices_qcomp] = max(plaid_responses_temp, [], 2);
% inputdatatemp{3}=abs(dirdiff(max_indices_qcomp));
% input_repr=Rat_repr_quasi{2};
% plaid_responses_temp = reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
% plaid_responses_temp = recenter_neuron_responses(plaid_responses_temp, 6);
% plaid_responses_temp = plaid_responses_temp(:, 13:24);
% [~, max_indices_qpatt] = max(plaid_responses_temp, [], 2);
% inputdatatemp{4}=abs(dirdiff(max_indices_qpatt));
% % plot max plaid dir on realigned matrices ----
% figure;
% distribtouse=inputdatatemp;
% inputpars.inputaxh=gca;
% hold(inputpars.inputaxh,'on')
% inputpars.boxplotwidth=0.4;%0.5;
% inputpars.boxplotlinewidth=2;
% inputpars.densityplotwidth=0.4;%0.5;
% inputpars.yimtouse=[0,180];
% inputpars.scatterjitter=inputpars.boxplotlinewidth*0.1;
% inputpars.scatteralpha=0.15;
% inputpars.scattersize=40;
% inputpars.distralpha=0.5;
% inputpars.xlabelstring=[];
% inputpars.ylabelstring='preferred dir';
% inputpars.titlestring=['preferred dir distr'];
% inputpars.boolscatteron=1;
% inputpars.ks_bandwidth=30;
% inputpars.xlimtouse=[-0.5,5.5];
% % plot violins
% inputadata.inputdistrs=distribtouse;
% inputpars.n_distribs=numel(inputadata.inputdistrs);
% inputpars.dirstrcenters=(1:inputpars.n_distribs);
% inputpars.xtickslabelvector={'comp','patt','quasi comp','quasi patt'};
% [~,scatter_xs,scatter_ys] = plot_violinplot_PN_new(inputadata,inputpars);
% xtickangle(45)

%% visualize grand averages for DorsalNet patt. and comp. predictions too ----------------

% initialize figure
fighand1quinqua = figure('units','normalized','outerposition',[0 0 1 1]);
% loop over populations (target)
for pop_idx1=1:(numel(pop_labels)-1)
    % loop over populations (predictor)
    for pop_idx2=1:(numel(pop_labels)-1)
        % get observed matrices
        input_repr=Rat_repr{pop_idx1};
        observed_mat_smooth=reorder_rows_by_peak_position(max_normalize_halves(gaussianSmooth1D(input_repr, smoothpar, 1)));
        % get predicted matrices
        predicted_mats_smooth=predicted_mats_smooths{pop_idx1,pop_idx2};
        predicted_mat_smooth=squeeze(nanmean(predicted_mats_smooth,1));
        % fetch input
        preprocessed_repr=predicted_mat_smooth;
        curr_neuron_responses=preprocessed_repr;
        curr_neuron_responses_ref=observed_mat_smooth;
        rearranged_responses = recenter_neuron_responses_ref(...
            curr_neuron_responses, curr_neuron_responses_ref, 6);
        rearranged_responses_avg = nanmean(rearranged_responses,1);
        rearranged_responses_se = nanstd(rearranged_responses,1)./sqrt(size(rearranged_responses,1));
        subplot(2,4,1+4*(pop_idx1-1)+2*(pop_idx2-1))
        imagesc(rearranged_responses); colormap(gray); caxis([0,1]);
        hold on;
        plot([6,6],[0.5,size(rearranged_responses,1)+0.5],'-','color',[inputpars.distrcolors{pop_idx1},0.5],'linewidth',2)
        plot([12+6,12+6],[0.5,size(rearranged_responses,1)+0.5],'-','color',[inputpars.distrcolors{pop_idx1},0.5],'linewidth',2)
        plot([12.5,12.5],[0.5,size(rearranged_responses,1)+0.5],'color',inputpars.distrcolors{pop_idx1},'linewidth',2)
        title([pop_labels{pop_idx1},' (',pop_labels{pop_idx2},'-pred) reord. repr-mat'],'color',inputpars.distrcolors{pop_idx1});
        ylabel('neuron #')
        xticks([1:3:numel(stimlabelstousebis)])
        xticklabels(stimlabelstousebis(1:3:end));
        xtickangle(45)
        set(gca,'fontsize',12)
        axis square
        subplot(2,4,2+4*(pop_idx1-1)+2*(pop_idx2-1))
        hold on;
        grat_tc_avg=rearranged_responses_avg(1:12);
        grat_tc_se=rearranged_responses_se(1:12);
        plaid_tc_avg=rearranged_responses_avg(13:24);
        plaid_tc_se=rearranged_responses_se(13:24);
        [ ~, ~, Zp, Zc, ~, ~, ~, ~, ~, ~ ] = ...
            get_pattern_index( grat_tc_avg',plaid_tc_avg' );
        alphatouse=0.15;
        xvals=(0:30:330)-30*5;
        colortouse=inputpars.distrcolors{pop_idx1};
        plot_shaded_mu_std(gca,xvals,grat_tc_avg,grat_tc_se,colortouse,alphatouse);
        plot(xvals,grat_tc_avg,'-','color',colortouse,'linewidth',2)
        colortouse=inputpars.distrcolors{pop_idx1}.*0.75;
        plot_shaded_mu_std(gca,xvals,plaid_tc_avg,plaid_tc_se,colortouse,alphatouse);
        text(-140,1,['Zc = ',num2str(round(Zc,2))],'fontsize',12)
        text(-140,0.95,['Zp = ',num2str(round(Zp,2))],'fontsize',12)
        text(-140,0.90,['PI = ',num2str(round(Zp-Zc,2))],'fontsize',12)
        plot(xvals,plaid_tc_avg,'-','color',colortouse,'linewidth',2)
        title([pop_labels{pop_idx1},' (',pop_labels{pop_idx2},'-pred) avg. tc'],'color',inputpars.distrcolors{pop_idx1});
        ylabel('normalized response')
        xlabel('direction')
        ylim([0.15,1.1])
        xlim([min(xvals),max(xvals)]);
        set(gca,'fontsize',12)
        plot([0,0],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
        plot([60,60],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
        plot([-60,-60],[0.15,1.1],'--','color',[inputpars.distrcolors{pop_idx1},0.25],'linewidth',2)
    end
end
sgtitle('DorsalNet-predicted patt. and comp. representations')
% save
saveas(fighand1quinqua,[resultpath,filesep,'DorsalNet_predicted_patt_comp_populations_diagnostics'],'jpg')
print(fighand1quinqua,'-depsc','-painters',[resultpath,filesep,['DorsalNet_predicted_patt_comp_populations_diagnostics'],'.eps'])
close all

%% plot cross-validated population representation bullet plot comparison ----------------

% initialize bullet comparison variables
Zc_obs=cell(2,1);
Zp_obs=cell(2,1);
Zc_pred_ln=cell(2,1);
Zp_pred_ln=cell(2,1);
Zc_pred_rpr=cell(2,2);
Zp_pred_rpr=cell(2,2);
ctlabel_pred_rpr=cell(2,2);
ctlabel_pred_ln=cell(2,1);
% organize data to get bullet plots
for pop_idx1=1:2
    % get indeces of valid neurons
    idx_valid_neu=Rat_rdm.bullet_distr.idx_consist{pop_idx1};
    % partial correlations
    Zc_obs{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zc_O_distribs{pop_idx1}(idx_valid_neu);
    Zp_obs{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zp_O_distribs{pop_idx1}(idx_valid_neu);
    Zc_pred_ln{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zc_P_distribs{pop_idx1}(idx_valid_neu);
    Zp_pred_ln{pop_idx1}=Rat_rdm.bullet_distr.collapsed_Zp_P_distribs{pop_idx1}(idx_valid_neu);
    % loop over types of representation prediction
    for pop_idx2=1:2
        current_pred_rprs=predicted_mats_smooths{pop_idx1,pop_idx2};
        current_pred_rpr=squeeze(nanmean(current_pred_rprs,1));
        % initialize Zp and Zc storage
        Zc_pred_rpr{pop_idx1,pop_idx2}=NaN(size(current_pred_rpr,1),1);
        Zp_pred_rpr{pop_idx1,pop_idx2}=NaN(size(current_pred_rpr,1),1);
        % initializeclassification storage
        ctlabel_pred_rpr{pop_idx1,pop_idx2}=NaN(size(current_pred_rpr,1),1);
        ctlabel_pred_ln{pop_idx1}=NaN(size(current_pred_rpr,1),1);
        for neu_idx=1:size(current_pred_rpr,1)
            % perform partial correlation analysis
            curr_grat_tc=current_pred_rpr(neu_idx,1:12);
            curr_plaid_tc=current_pred_rpr(neu_idx,12+(1:12));
            [ ~, ~, temp_Zp, temp_Zc, ~, ~, ~, ~, ~, ~ ] =...
                get_pattern_index( curr_grat_tc,curr_plaid_tc );
            % store results
            Zc_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=temp_Zc;
            Zp_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=temp_Zp;
            % store predicted classifications - rpr
            if temp_Zp-max(temp_Zc,0)>=1.28
                ctlabel_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=2; % 2=pattern
            elseif temp_Zc-max(temp_Zp,0)>=1.28
                ctlabel_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=1; % 1=component
            else
                ctlabel_pred_rpr{pop_idx1,pop_idx2}(neu_idx)=0; % 0=unclassified
            end
            % store predicted classifications - ln
            if Zp_pred_ln{pop_idx1}(neu_idx)-max(Zc_pred_ln{pop_idx1}(neu_idx),0)>=1.28
                ctlabel_pred_ln{pop_idx1}(neu_idx)=2; % 2=pattern
            elseif Zc_pred_ln{pop_idx1}(neu_idx)-max(Zp_pred_ln{pop_idx1}(neu_idx),0)>=1.28
                ctlabel_pred_ln{pop_idx1}(neu_idx)=1; % 1=component
            else
                ctlabel_pred_ln{pop_idx1}(neu_idx)=0; % 0=unclassified
            end
        end
    end
end
% initialize figure
fighand2 = figure('units','normalized','outerposition',[0 0 1 1]);
% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
% plot bullet plot observed -----------------------------------------------------
subplot(1,3,1)
hold on;
% loop over neurons
for pop_idx1=1:2
    for current_n=1:numel(Zc_obs{pop_idx1})
        % get data to plot
        Zc_O=Zc_obs{pop_idx1}(current_n);
        Zp_O=Zp_obs{pop_idx1}(current_n);
        Zc_P=Zc_pred_ln{pop_idx1}(current_n);
        Zp_P=Zp_pred_ln{pop_idx1}(current_n);
        % plot in bullet
        scatter(Zc_O,Zp_O,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        scatter(Zc_P,Zp_P,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{pop_idx1}.*0.75,...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        startvec=[Zc_O,Zp_O];
        endvec=[Zc_P,Zp_P];
        plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{pop_idx1}*0.75,0.25])
        
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
title(['Zp vs. Zc ','all layers',' obs. vs. ln-pred.']);
set(gca,'fontsize',12)
% plot bullet plot pred rpr (component) -----------------------------------------------------
subplot(1,3,2)
hold on;
% loop over neurons
for pop_idx1=1:2
    for current_n=1:numel(Zc_obs{pop_idx1})
        % get data to plot
        Zc_O=Zc_obs{pop_idx1}(current_n);
        Zp_O=Zp_obs{pop_idx1}(current_n);
        Zc_P=Zc_pred_rpr{pop_idx1,1}(current_n);
        Zp_P=Zp_pred_rpr{pop_idx1,1}(current_n);
        % plot in bullet
        scatter(Zc_O,Zp_O,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        scatter(Zc_P,Zp_P,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{pop_idx1}.*0.75,...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        startvec=[Zc_O,Zp_O];
        endvec=[Zc_P,Zp_P];
        plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{pop_idx1}*0.75,0.25])
        
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
title(['Zp vs. Zc ','all layers',' obs. vs. rpr-pred. (comp)']);
set(gca,'fontsize',12)
% plot bullet plot pred rpr (pattern) -----------------------------------------------------
subplot(1,3,3)
hold on;
% loop over neurons
for pop_idx1=1:2
    for current_n=1:numel(Zc_obs{pop_idx1})
        % get data to plot
        Zc_O=Zc_obs{pop_idx1}(current_n);
        Zp_O=Zp_obs{pop_idx1}(current_n);
        Zc_P=Zc_pred_rpr{pop_idx1,2}(current_n);
        Zp_P=Zp_pred_rpr{pop_idx1,2}(current_n);
        % plot in bullet
        scatter(Zc_O,Zp_O,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        scatter(Zc_P,Zp_P,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{pop_idx1}.*0.75,...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        startvec=[Zc_O,Zp_O];
        endvec=[Zc_P,Zp_P];
        plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{pop_idx1}*0.75,0.25])
        
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
title(['Zp vs. Zc ','all layers',' obs. vs. rpr-pred. (patt)']);
set(gca,'fontsize',12)
% add suptitle
suptitle('Rat - bullet plot - pattern and components - obs. vs pred.')
% save
saveas(fighand2,[resultpath,filesep,'Rat_vs_DorsalNet_bullet_prediction_comparison'],'jpg')
print(fighand2,'-depsc','-painters',[resultpath,filesep,['Rat_vs_DorsalNet_bullet_prediction_comparison'],'.eps'])

%% plot predicted population representation bullet plot comparison (best) ----------------

% initialize figure
fighand3 = figure('units','normalized','outerposition',[0 0 1 1]);
% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;
% plot bullet plot observed -----------------------------------------------------
subplot(1,2,1)
hold on;
% loop over neurons
for pop_idx1=1:2
    for current_n=1:numel(Zc_obs{pop_idx1})
        % get data to plot
        Zc_O=Zc_obs{pop_idx1}(current_n);
        Zp_O=Zp_obs{pop_idx1}(current_n);
        Zc_P=Zc_pred_ln{pop_idx1}(current_n);
        Zp_P=Zp_pred_ln{pop_idx1}(current_n);
        % plot in bullet
        scatter(Zc_O,Zp_O,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        scatter(Zc_P,Zp_P,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{pop_idx1}.*0.75,...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        startvec=[Zc_O,Zp_O];
        endvec=[Zc_P,Zp_P];
        plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{pop_idx1}*0.75,0.25])
        
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
title(['Zp vs. Zc ','all layers',' obs. vs. ln-pred.']);
set(gca,'fontsize',12)
% plot bullet plot pred rpr (matched) -----------------------------------------------------
subplot(1,2,2)
hold on;
% loop over neurons
for pop_idx1=1:2
    for current_n=1:numel(Zc_obs{pop_idx1})
        % get data to plot
        Zc_O=Zc_obs{pop_idx1}(current_n);
        Zp_O=Zp_obs{pop_idx1}(current_n);
        Zc_P=Zc_pred_rpr{pop_idx1,pop_idx1}(current_n);
        Zp_P=Zp_pred_rpr{pop_idx1,pop_idx1}(current_n);
        % plot in bullet
        scatter(Zc_O,Zp_O,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1},'MarkerEdgeColor',inputpars.distrcolors{pop_idx1},...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        scatter(Zc_P,Zp_P,50,...
            'MarkerFaceColor',inputpars.distrcolors{pop_idx1}.*0.75,'MarkerEdgeColor',inputpars.distrcolors{pop_idx1}.*0.75,...
            'MarkerFaceAlpha',1,'MarkerEdgeAlpha',1);
        startvec=[Zc_O,Zp_O];
        endvec=[Zc_P,Zp_P];
        plot([startvec(1),endvec(1)],[startvec(2),endvec(2)],'linewidth',2,'Color',[inputpars.distrcolors{pop_idx1}*0.75,0.25])
        
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
title(['Zp vs. Zc ','all layers',' obs. vs. rpr-pred. (matched)']);
set(gca,'fontsize',12)
% add suptitle
suptitle('Rat - bullet plot - pattern and components - obs. vs pred.')
% save
saveas(fighand3,[resultpath,filesep,'Rat_vs_DorsalNet_bullet_prediction_comparison_matched'],'jpg')
print(fighand3,'-depsc','-painters',[resultpath,filesep,['Rat_vs_DorsalNet_bullet_prediction_comparison_matched'],'.eps'])

%% plot proportion of neurons changing or keeping classification on predictions

% set celltypecodes
celltypecode=[1,2,0];
celltypelabel={'component','pattern','unclassified'};
changenums=cell(1,2);
changefracs=cell(1,2);
% organize data to get bullet plots
for pop_idx2=1:2
    changenums{pop_idx2}=NaN(2,3);
    changefracs{pop_idx2}=NaN(2,3);
    for pop_idx1=1:2
        % set current celltypecode
        curr_celltypecode=celltypecode(pop_idx1);
        if curr_celltypecode==1
            curr_celltypecode_other=2;
        else
            curr_celltypecode_other=1;
        end
        % count class changes
        curr_num_unchanged=sum(ctlabel_pred_rpr{pop_idx1,pop_idx2}==curr_celltypecode);
        curr_num_changed=sum(ctlabel_pred_rpr{pop_idx1,pop_idx2}==curr_celltypecode_other);
        curr_num_tounclass=sum(ctlabel_pred_rpr{pop_idx1,pop_idx2}==celltypecode(3));
        curr_num_tot=numel(ctlabel_pred_rpr{pop_idx1,pop_idx2});
        changenums{pop_idx2}(pop_idx1,:)=[curr_num_unchanged,curr_num_changed,curr_num_tounclass];
        changefracs{pop_idx2}(pop_idx1,:)=[curr_num_unchanged,curr_num_changed,curr_num_tounclass]./curr_num_tot;
    end
end

% initialize figure
fighand3bis = figure('units','normalized','outerposition',[0 0 1 1]);
for pop_idx2=1:2
    subplot(1,2,pop_idx2)
    hold on;
    % fetch data
    curr_changenums=changenums{pop_idx2};
    curr_changefracs=changefracs{pop_idx2};
    for pop_idx1=1:2
        if pop_idx1==1 %comp;
            colstouse={[50,200,0]./255,[255,150,0]./255,[0.5,0.5,0.5]};
        elseif pop_idx1==2 %patt;
            colstouse={[255,150,0]./255,[50,200,0]./255,[0.5,0.5,0.5]};
        end
        xvals=( 1:3 )+ 4*(pop_idx1-1);
        yvals=curr_changefracs(pop_idx1,:);
        yvalstext=curr_changenums(pop_idx1,:);
        for jj=1:numel(xvals)
            bar(xvals(jj),yvals(jj),'facecolor',colstouse{jj},'edgecolor',colstouse{jj},'barwidth',0.75);
            text(xvals(jj)-0.15,yvals(jj)+0.01,['n=',num2str(yvalstext(jj))],'color',colstouse{jj},'fontweight','bold','fontsize',9)
        end
    end
    xlim([0,8])
    ylim([0,0.9])
    set(gca,'fontsize',12)
    xticks([1:3,5:7]);
    xtickangle(45)
    xticklabels({'comp. same','comp. diff','comp. uncl','patt. same','patt. diff','patt. uncl'});
    ylabel('outcome fraction');
    plot([4,4],[0,1],'--','color',[0.5,0.5,0.5],'linewidth',2)
    title([celltypelabel{pop_idx2},'-prediction-based reclassification outcomes'])
    cm=cell2mat({[50,200,0]./255;[50,200,0]./255;[50,200,0]./255;[255,150,0]./255;...
        [255,150,0]./255;[255,150,0]./255});
    ax=gca;
    for itick = 1:numel(ax.XTickLabel)
        ax.XTickLabel{itick} = ...
            sprintf('\\color[rgb]{%f,%f,%f}%s',...
            cm(itick,:)*0.85,...
            ax.XTickLabel{itick});
    end
end
suptitle('DorsalNet prediction-based reclassification')
% save resuts
saveas(fighand3bis,[resultpath,filesep,'DN_prediction_based_reclassification_outcome'],'jpg')
print(fighand3bis,'-depsc','-painters',[[resultpath,filesep,'DN_prediction_based_reclassification_outcome'],'.eps'])

%% plot population representation predicted vs. observed Zc Zp violin plots (matched) ----------------

% initialize figure
fighand4 = figure('units','normalized','outerposition',[0 0 1 1]);
% set whether to use paired or unpaired testing
pairedtestingbool=1;
% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;

% plot Zc violin plot -----------------------------------------------------
subplot(1,2,1)
hold on;
distrtoplotlist{1}={...
    [Zc_obs{1}],...
    [Zc_pred_rpr{1,1}],...
    [Zc_obs{2}],...
    [Zc_pred_rpr{2,2}]};
ylabellist{1}='Zc';
yimtouselist{1}=[-4,11];
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
        ' - patt #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred (DN-matched)','pattern - pred (DN-matched)'};
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
    text(-0.25,13.5-4,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8-4,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1-4,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
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
distrtoplotlist{1}={...
    [Zp_obs{1}],...
    [Zp_pred_rpr{1,1}],...
    [Zp_obs{2}],...
    [Zp_pred_rpr{2,2}]};
ylabellist{1}='Zp';
yimtouselist{1}=[-4,11];
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
        ' - patt #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred (DN-matched)','pattern - pred (DN-matched)'};
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
    text(-0.25,13.5-4,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8-4,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1-4,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
hold on;
plot([scatter_xs{1},scatter_xs{2}]',[scatter_ys{1},scatter_ys{2}]','linewidth',2,'Color',[inputpars.distrcolors{1}*0.75,0.15])
plot([scatter_xs{3},scatter_xs{4}]',[scatter_ys{3},scatter_ys{4}]','linewidth',2,'Color',[inputpars.distrcolors{3}*0.75,0.15])
line([-0.5,5.5],[0,0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% add suptitle
sgtitle('Zp and Zc obs. pred. (DorsalNet-matched) comparison')
% save resuts
saveas(fighand4,[resultpath,filesep,'Zp_Zc_scatter_all_obs_vs_pred_DN_matched'],'jpg')
print(fighand4,'-depsc','-painters',[[resultpath,filesep,'Zp_Zc_scatter_all_obs_vs_pred_DN_matched'],'.eps'])

%% plot population representation predicted vs. observed Zc Zp violin plots (component) ----------------

% initialize figure
fighand5 = figure('units','normalized','outerposition',[0 0 1 1]);
% set whether to use paired or unpaired testing
pairedtestingbool=1;
% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;

% plot Zc violin plot -----------------------------------------------------
subplot(1,2,1)
hold on;
distrtoplotlist{1}={...
    [Zc_obs{1}],...
    [Zc_pred_rpr{1,1}],...
    [Zc_obs{2}],...
    [Zc_pred_rpr{2,1}]};
ylabellist{1}='Zc';
yimtouselist{1}=[-4,11];
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
        ' - patt #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred (DN-comp)','pattern - pred (DN-comp)'};
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
    text(-0.25,13.5-4,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8-4,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1-4,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
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
distrtoplotlist{1}={...
    [Zp_obs{1}],...
    [Zp_pred_rpr{1,1}],...
    [Zp_obs{2}],...
    [Zp_pred_rpr{2,1}]};
ylabellist{1}='Zp';
yimtouselist{1}=[-4,11];
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
        ' - patt #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred (DN-comp)','pattern - pred (DN-comp)'};
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
    text(-0.25,13.5-4,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8-4,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1-4,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
hold on;
plot([scatter_xs{1},scatter_xs{2}]',[scatter_ys{1},scatter_ys{2}]','linewidth',2,'Color',[inputpars.distrcolors{1}*0.75,0.15])
plot([scatter_xs{3},scatter_xs{4}]',[scatter_ys{3},scatter_ys{4}]','linewidth',2,'Color',[inputpars.distrcolors{3}*0.75,0.15])
line([-0.5,5.5],[0,0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% add suptitle
sgtitle('Zp and Zc obs. pred. (DorsalNet-comp) comparison')
% save resuts
saveas(fighand5,[resultpath,filesep,'Zp_Zc_scatter_all_obs_vs_pred_DN_comp'],'jpg')
print(fighand5,'-depsc','-painters',[[resultpath,filesep,'Zp_Zc_scatter_all_obs_vs_pred_DN_comp'],'.eps'])

%% plot population representation predicted vs. observed Zc Zp violin plots (pattern) ----------------

% initialize figure
fighand6 = figure('units','normalized','outerposition',[0 0 1 1]);
% set whether to use paired or unpaired testing
pairedtestingbool=1;
% get colors
inputpars.distrcolors{3}=[0,0,0];
inputpars.distrcolors{1}=[50,200,0]./255;
inputpars.distrcolors{2}=[255,150,0]./255;

% plot Zc violin plot -----------------------------------------------------
subplot(1,2,1)
hold on;
distrtoplotlist{1}={...
    [Zc_obs{1}],...
    [Zc_pred_rpr{1,2}],...
    [Zc_obs{2}],...
    [Zc_pred_rpr{2,2}]};
ylabellist{1}='Zc';
yimtouselist{1}=[-4,11];
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
        ' - patt #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred (DN-patt)','pattern - pred (DN-patt)'};
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
    text(-0.25,13.5-4,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8-4,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1-4,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
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
distrtoplotlist{1}={...
    [Zp_obs{1}],...
    [Zp_pred_rpr{1,2}],...
    [Zp_obs{2}],...
    [Zp_pred_rpr{2,2}]};
ylabellist{1}='Zp';
yimtouselist{1}=[-4,11];
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
        ' - patt #  = ',num2str(numel(distribtouse{3}),'%.0f'),' )'];
    inputpars.boolscatteron=1;
    inputpars.ks_bandwidth=ks_ban{jj};
    inputpars.xlimtouse=[-0.5,5.5];
    % plot violins
    inputadata.inputdistrs=distribtouse;
    inputpars.n_distribs=numel(inputadata.inputdistrs);
    inputpars.dirstrcenters=(1:inputpars.n_distribs);
    inputpars.xtickslabelvector={'component - obs ','pattern - obs','component - pred (DN-patt)','pattern - pred (DN-patt)'};
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
    text(-0.25,13.5-4,['comp median diff p = ',num2str(pvalw_comp)],'fontsize',11)
    text(-0.25,12.8-4,['patt median diff p = ',num2str(pvalw_patt)],'fontsize',11)
    text(-0.25,12.1-4,['patt vs. comp delta median diff p = ',num2str(pvalw_displ)],'fontsize',11)
    xtickangle(45)
    set(gca,'fontsize',12)
    axis square
end
hold on;
plot([scatter_xs{1},scatter_xs{2}]',[scatter_ys{1},scatter_ys{2}]','linewidth',2,'Color',[inputpars.distrcolors{1}*0.75,0.15])
plot([scatter_xs{3},scatter_xs{4}]',[scatter_ys{3},scatter_ys{4}]','linewidth',2,'Color',[inputpars.distrcolors{3}*0.75,0.15])
line([-0.5,5.5],[0,0],'LineWidth',1.5,'Color',[0.5,0.5,0.5]);
% add suptitle
sgtitle('Zp and Zc obs. pred. (DorsalNet-patt) comparison')
% save resuts
saveas(fighand6,[resultpath,filesep,'Zp_Zc_scatter_all_obs_vs_pred_DN_patt'],'jpg')
print(fighand6,'-depsc','-painters',[[resultpath,filesep,'Zp_Zc_scatter_all_obs_vs_pred_DN_patt'],'.eps'])

%% ------------------------------------------------------------------------

% % NB: overfitting is easy ...
% target_mat = Rat_repr{2};
% predictor_mat = DorsalNet_repr{2};
% regpar = 0.0001;
% [fitted_coeffs, fitted_cost] = predict_representation(target_mat, predictor_mat, regpar);
% predicted_mat = linear_model(predictor_mat,fitted_coeffs);
% figure; imagesc(target_mat)