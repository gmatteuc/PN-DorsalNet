function [] = compute_STA_DorsalNet_activations_PN( input_psthmat,inputpars )

% get input pars
output_folder=inputpars.output_folder;
noise_path=inputpars.noise_path;
% get input data
psth_by_frames = input_psthmat;

% set pars
pars = set_pars_PN();

% load noise stimulus
load(noise_path,'Noise');
rvideo_temp=Noise; clear Noise;
target_STAsize=[32 32];
rvideo=NaN([target_STAsize,size(rvideo_temp,3),size(rvideo_temp,1)]);
for ii=1:size(rvideo_temp,1)
    for kk=1:size(rvideo_temp,3)
        rvideo(:,:,kk,ii) = imresize(squeeze(rvideo_temp(ii,1,kk,:,:)),target_STAsize);
    end
end

% fetch STA pars
maxlag = pars.STA_depth;
fwidth = target_STAsize(2);
fheight = target_STAsize(1);
ridgeparam = pars.STA_ridgeparam;

% run analysis  ------------------------------

% loop over neurons
numstart=1;
Nneu=size(psth_by_frames,2);
for neuronum = numstart:Nneu
    
    % skip unit if already analyzed
    fnam=[output_folder,filesep,'unit_',num2str(neuronum),'_results'];
    if exist(fnam,'dir')==7
        fprintf(['unit ',num2str(neuronum),' already analyzed, skipping... \n'])
        numstart=neuronum+1;
    else
        
        if 1 % process good units only NB: modified 03/11/2022, for now do on all
            
            % initialize variables in witch store results at different lags
            Dstafr=zeros(fheight,fwidth,maxlag);
            Dwstafr=zeros(fheight,fwidth,maxlag);
            DZwstafr=zeros(fheight,fwidth,maxlag);
            DZstafr=zeros(fheight,fwidth,maxlag);
            Drawcov=zeros(fheight*fwidth,fheight*fwidth);
            Dneuronum=neuronum; %#ok<NASGU>
            
            % reformat stimulus movie
            Stim=zeros(size(rvideo,3)*size(rvideo,4),size(rvideo,2)*size(rvideo,1));
            for ll=1:size(rvideo,4)
                for k=1:size(rvideo,3)
                    frm=rvideo(:,:,k,ll);
                    % frames are transformed to columns, succensive rows represent successive frames
                    Stim(k+(ll-1)*size(rvideo,3),:)=frm(:);
                end
            end
            
            % loop pre-spike lags (along the depth of the filter)
            for nlag = 1:maxlag
                
                if nlag==1
                    % create output folder and cd in
                    fnam=[output_folder,'/neuron_',num2str(neuronum),'_medium_results'];
                    mkdir(fnam);
                    oldfold=cd(fnam);
                else
                end
                
                % reformat spike data into a single spike per frame vector aligned with stimulus movie
                sp=zeros(1,size(psth_by_frames,1)*size(psth_by_frames,3));
                hsp=zeros(1,size(psth_by_frames,3));
                for mm=1:size(psth_by_frames,1)
                    for ff=1:size(psth_by_frames,3)
                        sp(ff+(mm-1)*size(psth_by_frames,3))=psth_by_frames(mm,neuronum,ff);
                        hsp(ff)=hsp(ff)+psth_by_frames(mm,neuronum,ff);
                    end
                end
                totspikes=sum(hsp);
                message=['\nAnalyzing neuron ',num2str(neuronum),' ...\n'];
                fprintf(message)
                message=['total number of spikes = ',num2str(totspikes),'\n'];
                fprintf(message)
                % message=[num2str(fwidth*fheight*50),' spikes required for a good STC reconstruction according to Rust rule\n'];
                
                if nlag==1
                    Dtotspikes=totspikes; %#ok<NASGU>
                    % plot and save global PSTHs
                    f1=figure;
                    set(f1,'Position',[10,10,1500,1000]);
                    plot(hsp,'color',[0.1,0.3,0.9],'LineWidth',1.5)
                    hold on
                    plot(sgolayfilt(hsp, 2, 27),'k','LineWidth',3)
                    legend('Mean Noise PSTH','Mean Noise PSTH - Smoothed');
                    ylabel('spikes per frame');
                    xlabel('frame number');
                    title(['Neuron ',num2str(neuronum),': ',num2str(totspikes),' spikes before'])
                    hold off
                    filename1=[fnam,filesep,'global PSTH.jpg'];
                    set(gcf, 'PaperPositionMode', 'auto');
                    saveas(f1, filename1);
                    close(f1)
                else
                end
                
                %% perform STA analysis at current lag ---------------------------
                
                % set analysisis parameters
                CriticalSize = 1e8;
                n=1;
                % reshape sp ad appy lag
                sp = sp';
                sp = circshift(sp,-nlag);
                sp(end-nlag+1:end)=zeros(1,nlag);
                % standardize stimulus
                Stim=(Stim-mean(Stim(:)))*(std(Stim(:))).^-1;
                
                if nlag==1 && neuronum==numstart % do the work on the correlation matrix only once
                    
                    % compute stimulus ensemble covariance matrix
                    [~,~,~,rawcov] = simpleSTC( Stim,sp, n, CriticalSize);
                    
                    % perform Tikhonov regularized inversion of the covariance matrix
                    covInv = inv(rawcov+ridgeparam*eye(size(rawcov)));
                    covInvsqrt = sqrtm(covInv); %#ok<NASGU>
                    
                end
                
                % compute STA
                [sta] = simpleSTA(Stim, sp, n,CriticalSize);
                
                % whiten STA
                wsta = covInv*sta; %#ok<MINV>
                
                % reshape and visualize sta
                stafr=reshape(sta,fheight,fwidth);
                if nlag==3
                    f1=figure;
                    set(f1,'Position',[10,10,1500,1000]); imagesc(stafr); colormap('gray'); set(gca,'dataAspectRatio',[1 1 1]); colorbar; title('raw STA filter');
                    set(gcf, 'PaperPositionMode', 'auto');
                    filename1=[fnam,filesep,'raw STA.jpg'];
                    saveas(f1, filename1);
                    close(f1)
                else
                end
                
                % reshape and visualize wsta
                wstafr=reshape(wsta,fheight,fwidth);
                if nlag==3
                    f1=figure;
                    set(f1,'Position',[10,10,1500,1000]); imagesc(wstafr); colormap('gray'); set(gca,'dataAspectRatio',[1 1 1]); colorbar; title('whitened STA filter');
                    set(gcf, 'PaperPositionMode', 'auto');
                    filename1=[fnam,filesep,'whitened STA.jpg'];
                    saveas(f1, filename1);
                    close(f1)
                else
                end
                
                %% perform permutation test at current lag ---------------------------
                
                % initialize permuation test variables
                nperm=30;
                staT=zeros(size(sta,1),nperm);
                wstaT=zeros(size(wsta,1),nperm);
                length_spikes=length(sp);
                
                % loop over permutations
                for j=1:nperm
                    
                    % randomly reshuffle spike timestamps in time
                    spP=sp(randperm(length_spikes));
                    
                    % perform ST analysis again
                    spP = circshift(spP,-nlag);
                    spP(end-nlag+1:end)=zeros(1,nlag);
                    [staP] = simpleSTA( Stim, spP, n, CriticalSize);
                    
                    % whiten permuted STA
                    wstaP = covInv*staP; %#ok<MINV>
                    staT(:,j)=staP;
                    wstaT(:,j)=wstaP;
                    
                    message=['permutation number ',num2str(j),' completed\n'];
                    fprintf(message)
                end
                
                % compute permutation means
                musta=mean(staT,2);
                muwsta=mean(wstaT,2);
                
                % compute permuatation standard deviations
                sigsta=std(staT,0,2);
                sigwsta=std(wstaT,0,2);
                
                %% permutation test results visualization ------------------------------
                
                % reshape permutation results
                mustafr=reshape(musta,fheight,fwidth);
                muwstafr=reshape(muwsta,fheight,fwidth);
                
                % compute Z-scored STA, whitened STA and STC
                Zstafr = (stafr-mustafr)./mean(sigsta(:));
                Zwstafr = (wstafr-muwstafr)./mean(sigwsta(:));
                
                % visualize Z-scored STA, whitened STA and STC
                Z1=figure; set(Z1,'Position',[10,10,1500,1000]); set(gca,'dataAspectRatio',[1 1 1]);
                imagesc(Zstafr,[-8,8]); colormap('gray'); colorbar; title('Z-scored STA filter'); set(gcf, 'PaperPositionMode', 'auto');
                Z2=figure; set(Z2,'Position',[10,10,1500,1000]); set(gca,'dataAspectRatio',[1 1 1]);
                imagesc(Zwstafr,[-8,8]); colormap('gray'); colorbar; title('Z-scored whitened STA filter'); set(gcf, 'PaperPositionMode', 'auto');
                filename1=[fnam,filesep,'Z_scored_STA_at_lag',num2str(nlag),'.jpeg'];
                filename2=[fnam,filesep,'Z_scored_wSTA_at_lag',num2str(nlag),'.jpeg'];
                saveas(Z1, filename1);
                saveas(Z2, filename2);
                close(Z1)
                close(Z2)
                
                %% compute significance maps ------------------------------
                
                N=5; % confidence level for the map in sigmas
                
                % significativity for wSTA
                signifmapwsta=zeros(size(wsta));
                for i=1:size(wsta,1)
                    if abs(wsta(i)-muwsta(i))>N*sigwsta(i)
                        signifmapwsta(i)=abs(abs(wsta(i))-abs(muwsta(i)));
                    end
                end
                signifmapwstafr=reshape(signifmapwsta,fheight,fwidth);
                
                % significativity for STA
                signifmapsta=zeros(size(sta));
                for i=1:size(sta,1)
                    if abs(sta(i)-musta(i))>N*sigsta(i)
                        signifmapsta(i)=abs(abs(sta(i))-abs(musta(i)));
                    end
                end
                signifmapstafr=reshape(signifmapsta,fheight,fwidth);
                
                %% plot global analysis results ------------------------------
                
                imaghand=figure;
                set(imaghand,'Position',[10,10,1500,1000]);
                % raw results
                ax1=subplot(3,2,1);
                imagesc(wstafr); colormap(ax1,gray); set(gca,'dataAspectRatio',[1 1 1]); colorbar;
                caxis([-2*quantile(wstafr(:),0.98),2*quantile(wstafr(:),0.98)])
                title('reconstructed wSTA filter', 'FontSize', 12);
                set(gca,'fontsize',12)
                ax2=subplot(3,2,2);
                imagesc(stafr); colormap(ax2,gray); set(gca,'dataAspectRatio',[1 1 1]); colorbar;
                caxis([-2*quantile(stafr(:),0.98),2*quantile(stafr(:),0.98)])
                title('reconstructed STA filter', 'FontSize', 12);
                set(gca,'fontsize',12)
                % signifmaps
                ax3=subplot(3,2,3);
                imagesc(signifmapwstafr,[min(signifmapwstafr(:)),max(signifmapwstafr(:))+0.0001]); colormap(ax3,hot); set(gca,'dataAspectRatio',[1 1 1]); colorbar;
                title('signifmap wSTA filter', 'FontSize', 12);
                set(gca,'fontsize',12)
                ax4=subplot(3,2,4);
                imagesc(signifmapstafr,[min(signifmapstafr(:)),max(signifmapstafr(:))+0.0001]); colormap(ax4,hot); set(gca,'dataAspectRatio',[1 1 1]); colorbar;
                title('signifmap STA filter', 'FontSize', 12);
                set(gca,'fontsize',12)
                % raw permutation means
                ax5=subplot(3,2,5);
                imagesc(muwstafr,[min(muwstafr(:)),max(muwstafr(:))]); colormap(ax5,gray); set(gca,'dataAspectRatio',[1 1 1]); colorbar;
                title('wSTA permutation mean', 'FontSize', 12);
                set(gca,'fontsize',12)
                ax6=subplot(3,2,6);
                imagesc(mustafr,[min(mustafr(:)),max(mustafr(:))]); colormap(ax6,gray); set(gca,'dataAspectRatio',[1 1 1]); colorbar;
                title('STA permutation mean', 'FontSize', 12);
                set(gca,'fontsize',12)
                suptitle(['unit ',num2str(neuronum),' ',inputpars.current_layer_name,' - lag = ',num2str(nlag)])
                filenam=[fnam,filesep,'ST analysis result at lag',num2str(nlag),'.jpeg'];
                saveas(imaghand, filenam);
                
                %% save results ------------------------------
                
                % store results at current lag
                Dstafr(:,:,nlag)=stafr;
                Dwstafr(:,:,nlag)=wstafr;
                DZwstafr(:,:,nlag)=Zwstafr;
                DZstafr(:,:,nlag)=Zstafr;
                if nlag==1
                    Drawcov(:,:)=rawcov;
                end
                
                if nlag==maxlag
                    
                    % initialize video writer
                    v1 = VideoWriter([fnam,filesep,'neuron_',num2str(neuronum),'_sta_movie'],'MPEG-4'); %#ok<TNMLP>
                    v1.FrameRate = 5;
                    open(v1);
                    fff=figure;
                    caxlim1=1*max([abs(quantile(DZstafr(:),0.001)),abs(quantile(DZstafr(:),0.999))]);
                    caxlim2=1*max([abs(quantile(DZwstafr(:),0.001)),abs(quantile(DZstafr(:),0.999))]);
                    frameidx=fliplr(1:size(Dstafr,3));
                    % loop over frames
                    for iiii=1:size(Dstafr,3)
                        ggg1=subplot(2,1,1); %#ok<NASGU>
                        imagesc(DZstafr(:,:,frameidx(iiii))); colormap('gray'); colorbar; title(['Z-scored STA filter - frame = ',num2str(frameidx(iiii))]);
                        axis equal; caxis([-caxlim1,caxlim1]); xlim([0.5,32.5]);
                        ggg2=subplot(2,1,2); %#ok<NASGU>
                        imagesc(DZwstafr(:,:,frameidx(iiii))); colormap('gray'); colorbar; title(['Z-scored whitened STA filter - frame = ',num2str(frameidx(iiii))]);
                        axis equal; caxis([-caxlim2,caxlim2]); xlim([0.5,32.5]);
                        xlim
                        % append frame to video
                        frame = getframe(fff);
                        writeVideo(v1,frame)
                    end
                    % close video file
                    close(v1);
                    close all
                    
                    % save results
                    save([output_folder,filesep,'STA_results_neuron_',num2str(neuronum),'.mat'],'Dstafr','Dwstafr','DZwstafr','DZstafr','Drawcov','Dneuronum','Dtotspikes')
                    % cd back to original folder
                    cd(oldfold)
                else
                end
                
                close all
                
            end
        end
    end
end
end
% NB: derived from [] = spike_triggered_average_gaussian_PN_MEDIUM( sind, bind )