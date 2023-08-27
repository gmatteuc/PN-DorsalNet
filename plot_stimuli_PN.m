clear all
close all
clc

% set paths
stimulipath='D:\DorsalNet\stimuli';
% add paths
addpath(genpath(stimulipath));

% load data
load([stimulipath,filesep,'Stimuli_DorsalNet_Matlab2.mat']) % even if actually use is 3

% get example frames to plot
grating_example_frame=squeeze(Gratings(6,1,1,:,:));
plaid_example_frame=squeeze(Plaids(6,1,1,:,:));

% plot example frames
f1 = figure('units','normalized','outerposition',[0 0 1 1]);
ax1=gca;
imagesc(grating_example_frame); colormap(gray); axis square;
xticks([]); yticks([]);
axis off

f2 = figure('units','normalized','outerposition',[0 0 1 1]);
ax2=gca;
imagesc(ax2,plaid_example_frame); colormap(ax2,gray); axis square;
xticks([]); yticks([]);
axis off

% save figure
saveas(f1,[stimulipath,filesep,'example_frame_grating'],'jpg')
saveas(f2,[stimulipath,filesep,'example_frame_plaid'],'jpg')
print(f1,'-depsc','-painters',[[stimulipath,filesep,'example_frame_grating'],'.eps'])
print(f2,'-depsc','-painters',[[stimulipath,filesep,'example_frame_plaid'],'.eps'])

close all