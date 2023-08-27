% initialize bullet comparison variables
n_runs = 10000; % test 5 ---> 2000
n_random = 1000; % test 5 ---> 1000
shift=0;
% initialize results storage
fractpatt_random=NaN(1,n_runs);
fractcomp_random=NaN(1,n_runs);
fractuncl_random=NaN(1,n_runs);
% loop over runs
parfor i=1:n_runs
    tic
    % initialize storage structures for current run
    Zc_resps_random=NaN(1,n_random);
    Zp_resps_random=NaN(1,n_random);
    ctlabel_resps_random=NaN(n_random,1);
    % generate and preprocess random tuning curves
    resps_random=randn(n_random,24);
    resps_random=max_normalize_halves(resps_random);
    % simulate random tuning curves
    for neu_idx=1:size(resps_random,1)
        % perform partial correlation analysis
        curr_grat_tc=resps_random(neu_idx,1:12);
        curr_plaid_tc=resps_random(neu_idx,12+(1:12));
        [ ~, ~, temp_Zp, temp_Zc, ~, ~, ~, ~, ~, ~ ] =...
            get_pattern_index_shifted( curr_grat_tc,curr_plaid_tc,shift );
        % store results
        Zc_resps_random(neu_idx)=temp_Zc;
        Zp_resps_random(neu_idx)=temp_Zp;
        % store respsicted classifications - random
        if temp_Zp-max(temp_Zc,0)>=1.28 && max(abs([curr_grat_tc,curr_plaid_tc]))>=0
            ctlabel_resps_random(neu_idx)=2; % 2=pattern
        elseif temp_Zc-max(temp_Zp,0)>=1.28 && max(abs([curr_grat_tc,curr_plaid_tc]))>=0
            ctlabel_resps_random(neu_idx)=1; % 1=component
        elseif max([curr_grat_tc,curr_plaid_tc])>=1
            ctlabel_resps_random(neu_idx)=0; % 0=unclassified
        else
            ctlabel_resps_random(neu_idx)=NaN;
        end
    end
    % store classification results for current run
    fractpatt_random(i)=nansum(ctlabel_resps_random==2)./n_random;
    fractcomp_random(i)=nansum(ctlabel_resps_random==1)./n_random;
    fractuncl_random(i)=nansum(ctlabel_resps_random==0)./n_random;
    toc
end


nanmean(fractpatt_random)
nanmean(fractcomp_random)
nanmean(fractuncl_random)
quantile(fractpatt_random,0.95)
save('test6.mat','fractpatt_random','fractcomp_random','fractuncl_random')

outfold=['D:\Backups\Personal_bk\PN_acute_analysis\processed_data\rf_quality_comparison_2022_final'];
%%
fighand5bis=figure('units','normalized','outerposition',[0 0 1 1]); 
hold on;
[a,b]=hist(fractpatt_random,62); %#ok<HIST>
bar(b,a./sum(a),'Facecolor',[0,0,0],'Edgecolor',[0,0,0],'Facealpha',1,'Edgealpha',0)
p1=plot([quantile(fractpatt_random,0.95),quantile(fractpatt_random,0.95)],get(gca,'ylim'),'--','linewidth',3,'color',[0.5,0.5,0.5]);
p2=plot([pattfraction_shifts(shifts==0),pattfraction_shifts(shifts==0)],get(gca,'ylim'),'-','linewidth',3,'color',inputpars.distrcolors{2});
legend([p1,p2],{'0.95-quantile','data'})
xlabel('frac pattern')
ylabel('relative frequency')
set(gca,'fontsize',12)
title('fraction of pattern distribution (random tc vs. data)')
saveas(fighand5bis,[outfold,filesep,'fraction_pattern_random_tc_vs_observed_distr'],'jpg')
print(fighand5bis,'-depsc','-painters',[[outfold,filesep,'fraction_pattern_random_tc_vs_observed_distr'],'.eps'])


%% control analysis from (overall_patt_comp_prediction_analysis_PN)

% % % get overall cell distribution
% % % obs_comp_rep_mat=max_normalize_halves(gaussianSmooth1D(obs_COUNT_per_class{1}, smoothpar, 1));
% % % obs_patt_rep_mat=max_normalize_halves(gaussianSmooth1D(obs_COUNT_per_class{2}, smoothpar, 1));
% % % obs_uncl_mat=max_normalize_halves(gaussianSmooth1D(obs_COUNT_per_class{3}, smoothpar, 1));
% % % obs_all_rep_mat=cat(1,cat(1,cat(1,obs_comp_rep_mat),obs_patt_rep_mat),obs_uncl_mat);
% % 
% % % obs_comp_rep_mat=max_normalize_halves(obs_COUNT_per_class{1});
% % % obs_patt_rep_mat=max_normalize_halves(obs_COUNT_per_class{2});
% % % obs_uncl_mat=max_normalize_halves(obs_COUNT_per_class{3});
% % % obs_all_rep_mat=cat(1,cat(1,cat(1,obs_comp_rep_mat),obs_patt_rep_mat),obs_uncl_mat);
% % 
% % 
% % obs_patt_rep_mat=max_normalize_halves(gaussianSmooth1D(obs_COUNT_per_class{2}, smoothpar, 1));
% % obs_uncl_mat=max_normalize_halves(gaussianSmooth1D(obs_COUNT_per_class{3}, smoothpar, 1));
% % obs_all_rep_mat=cat(1,obs_patt_rep_mat,obs_uncl_mat);
% % 
% % 
% % shifts=-7:7;
% % Zc_shift=cell(1,numel(shifts));
% % Zp_shift=cell(1,numel(shifts));
% % ctlabel_shift=cell(1,numel(shifts));
% % % organize data to get bullet plots
% % for shift_idx=1:numel(shifts)
% %     for neu_idx=1:size(obs_all_rep_mat,1)
% %         % perform partial correlation analysis
% %         curr_grat_tc=obs_all_rep_mat(neu_idx,1:12);
% %         curr_plaid_tc=obs_all_rep_mat(neu_idx,12+(1:12));
% %         [ ~, ~, temp_Zp, temp_Zc, ~, ~, ~, ~, ~, ~ ] =...
% %             get_pattern_index_shifted( curr_grat_tc,curr_plaid_tc,shifts(shift_idx) );
% %         % store results
% %         Zc_shift{shift_idx}(neu_idx)=temp_Zc;
% %         Zp_shift{shift_idx}(neu_idx)=temp_Zp;
% %         % store predicted classifications - rpr
% %         if temp_Zp-max(temp_Zc,0)>=1.28
% %             ctlabel_shift{shift_idx}(neu_idx)=2; % 2=pattern
% %         elseif temp_Zc-max(temp_Zp,0)>=1.28
% %             ctlabel_shift{shift_idx}(neu_idx)=1; % 1=component
% %         else
% %             ctlabel_shift{shift_idx}(neu_idx)=0; % 0=unclassified
% %         end
% %     end
% % end
% % pattfraction_shifts=NaN(1,numel(shifts));
% % for shift_idx=1:numel(shifts)
% % pattfraction_shifts(shift_idx)=nansum(ctlabel_shift{shift_idx}==2)./size(obs_all_rep_mat,1);
% % end
% % 
% % fighand5=figure('units','normalized','outerposition',[0 0 1 1]);
% % % subplot(1,3,[1,2])
% % hold on;
% % plot([min(shifts),max(shifts)],[quantile(fractpatt_random,0.95),quantile(fractpatt_random,0.95)],'--','color',[0,0,0])
% % plot([min(shifts),max(shifts)],[quantile(fractpatt_random,0.05),quantile(fractpatt_random,0.05)],'--','color',[0,0,0])
% % plot([min(shifts),max(shifts)],[nanmean(fractpatt_random),nanmean(fractpatt_random)],'-','linewidth',3,'color',[0,0,0])
% % plot(shifts,pattfraction_shifts,'-','linewidth',3,'color',inputpars.distrcolors{2})
% % scatter(0,pattfraction_shifts(shifts==0),155,...
% %     'Markerfacecolor',inputpars.distrcolors{2}*0,...
% %     'Markeredgecolor',inputpars.distrcolors{2}*0)
% % pval=1-compute_quantile(fractpatt_random, pattfraction_shifts(shifts==0));
% % scatter(shifts,pattfraction_shifts,45,...
% %     'Markerfacecolor',inputpars.distrcolors{2},...
% %     'Markeredgecolor',inputpars.distrcolors{2})
% % text(0.5,0.1125,['p perm = ',num2str(pval)],'fontsize',12)
% % text(0.5,0.11,['frac patt null = ',num2str(round(nanmean(fractpatt_random),4))],'fontsize',12)
% % text(0.5,0.1075,['frac patt obs = ',num2str(round(pattfraction_shifts(shifts==0),4))],'fontsize',12)
% % xlabel('Zp target dir shift')
% % ylabel('frac pattern')
% % xlim([min(shifts),max(shifts)])
% % set(gca,'fontsize',12)
% % title('fraction of pattern (random tc vs. observed)')
% % axis square
% % % subplot(1,3,3)
% % % hold on;
% % % [a,b]=hist(fractpatt_random,62); %#ok<HIST>
% % % bar(b,a./sum(a),'Facecolor',[0,0,0],'Edgecolor',[0,0,0],'Facealpha',1,'Edgealpha',0)
% % % p1=plot([quantile(fractpatt_random,0.95),quantile(fractpatt_random,0.95)],get(gca,'ylim'),'--','linewidth',2,'color',[0.5,0.5,0.5]);
% % % p2=plot([pattfraction_shifts(shifts==0),pattfraction_shifts(shifts==0)],get(gca,'ylim'),'-','linewidth',2,'color',inputpars.distrcolors{2});
% % % legend([p1,p2],{'0.95-quantile','data'})
% % % xlabel('frequency')
% % % ylabel('frac pattern')
% % % set(gca,'fontsize',12)
% % % title('fraction of pattern distribution (random tc vs. data)')
% % % axis square
% % saveas(fighand5,[outfold,filesep,'fraction_pattern_random_tc_vs_observed'],'jpg')
% % print(fighand5,'-depsc','-painters',[[outfold,filesep,'fraction_pattern_random_tc_vs_observed'],'.eps'])

