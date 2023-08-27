function pars = set_pars_PN()

% pars = set_pars_PN()
% set analysis paths and parameters along with stimuli and sessions metadata
% -------------------------------------------------------------------------

%% initialize pars structure

pars = struct();

%% set paths

current_dir='D:\Backups\Personal_bk\PN_acute_analysis';
% if not(isunix)
% current_dir=split(current_dir,'\PN_analysis');
% current_dir=current_dir{1};
% pars.raw_data_folder = [current_dir,'\PN_analysis\raw_data'];
% pars.processed_data_folder = [current_dir,'\PN_analysis\processed_data'];
% pars.LFPprocessed_data_folder = [current_dir,'\PN_analysis\processed_data\LFP_analysis'];
% pars.code_folder = genpath([current_dir,'\PN_analysis\functions']);
% pars.stim_folder = [current_dir,'\PN_analysis\stims'];
% pars.bitcode_folder = [current_dir,'\PN_analysis\bitcodes'];
% pars.LFPcode_folder = [current_dir,'\PN_analysis\functions\LFP_analysis'];
% pars.NAS_folder = '\\zocconasphys3.cns.sissa.it\Non-Linear_Visual_Processing\Data\PN_recordings';
% else
% current_dir=strrep(current_dir,'/PN_analysis/scripts','');
% pars.raw_data_folder = [current_dir,'/PN_analysis/raw_data'];
% pars.processed_data_folder = [current_dir,'/PN_analysis/processed_data'];
% pars.LFPprocessed_data_folder = [current_dir,'/PN_analysis/processed_data/LFP_analysis'];
% pars.code_folder = genpath([current_dir,'/PN_analysis/functions']);
% pars.stim_folder = [current_dir,'/PN_analysis/stims'];
% pars.bitcode_folder = [current_dir,'/PN_analysis/bitcodes'];
% pars.LFPcode_folder = [current_dir,'/PN_analysis/functions/LFP_analysis'];
% pars.NAS_folder = '//zocconasphys3.cns.sissa.it/Non-Linear_Visual_Processing/Data/PN_recordings';
% end

pars.raw_data_folder = [current_dir,'\raw_data'];
pars.processed_data_folder = [current_dir,'\processed_data'];
pars.LFPprocessed_data_folder = [current_dir,'\processed_data\LFP_analysis'];
pars.code_folder = genpath([current_dir,'\functions']);
pars.stim_folder = [current_dir,'\stims'];
pars.bitcode_folder = [current_dir,'\bitcodes'];
pars.LFPcode_folder = [current_dir,'\functions\LFP_analysis'];
pars.NAS_folder = '\\zocconasphys3.cns.sissa.it\Non-Linear_Visual_Processing\Data\PN_recordings';

%% list of sessions/blocks and their type

templist={'23_11_2016','02_03_2017','16_03_2017','17_03_2017','21_03_2017',...
    '22_03_2017','27_03_2017','04_05_2017','09_05_2017','16_05_2017',...
    '04_09_2017','12_09_2017','14_09_2017','18_09_2017','20_09_2017',...
    '26_09_2017','29_09_2017','08_11_2017','13_11_2017','28_11_2017','19_12_2017',...
    '13_02_2018','20_02_2018','06_03_2018','15_03_2018','22_03_2018','12_04_2018',...
    '19_04_2018','03_05_2018','09_05_2018'};

listSessions = cell(9,size(templist,2));
listSessions(1,:) = templist;

listSessions(2,:) = {[3,5],[2,6],[4,7],[5,9],[2,8],[2,5],...
    [3,9],[2,6],[2,6],[7],[5,10],[3,6],[7],[4,8],[3],[8],[4,6],[4],[4],[5],[7],...
    [6,10],[5],[5,8],[3],[3],[6],[4],[3],[4,11]};                                               % block numbers
listSessions(3,:) = {[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],...
    [1,1],[1,1],[1,1],[1],[1,1],[1,1],[1],[1,1],[1],[1],[1,1],[1],[1],[1],[1],...
    [1,1],[1],[1,1],[1],[1],[1],[1],[1],[1,1]};                                                 % blocktype (1=main protocol, 0=RF mapping)
listSessions(4,:) = {[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0], ...
    [0,0],[0],[2,1],[2,1],[1],[2,1],[1],[0],[2,2],[1],[2],[2],[0],...
    [2,0],[0],[1,0],[1],[1],[1],[1],[2],[2,1]};                                                 % targeted area (0=V1 fentanest, 1=LM fentanest, 2=RL fentanest, 3=AL fentanest)
listSessions(5,:) = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};              % recording type (1=head-fixed,0=acute)
listSessions(6,:) = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};              % targeted depth (1=horizontal penetration,0=vertical penetration)

listSessions(7,:) = {[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200)]...
    [2*ones(1,200),ones(1,200)],[ones(1,200)],[ones(1,200)],...
    [2*ones(1,200),ones(1,200)],[2*ones(1,200)],...
    [zeros(1,200)],[2*ones(1,200),2*ones(1,200)],...
    [1*ones(1,200)],[2*ones(1,200)],[2*ones(1,200)],[zeros(1,200)],...
    [2*ones(1,200),0*ones(1,200)],[0*ones(1,200)],[1*ones(1,200),0*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[2*ones(1,200)],[2*ones(1,200),1*ones(1,200)]...    
    };      % area manual annotation - before first main block - (0=V1 fentanest, 1=LM fentanest, 2=RL fentanest, 3=AL fentanest)
listSessions(8,:) = {[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200)]...
    [2*ones(1,200),ones(1,200)],[ones(1,200)],[ones(1,200)],...
    [2*ones(1,200),ones(1,200)],[2*ones(1,200)],...
    [zeros(1,200)],[2*ones(1,200),2*ones(1,200)],...
    [1*ones(1,200)],[2*ones(1,200)],[2*ones(1,200)],[zeros(1,200)],...
    [2*ones(1,200),0*ones(1,200)],[0*ones(1,200)],[1*ones(1,200),0*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[2*ones(1,200)],[2*ones(1,200),1*ones(1,200)]...
    };      % area manual annotation - before second main block - (0=V1 fentanest, 1=LM fentanest, 2=RL fentanest, 3=AL fentanest)
listSessions(9,:) = {[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],...
    [zeros(1,200),zeros(1,200)],[zeros(1,200),zeros(1,200)],[zeros(1,200)]...
    [2*ones(1,200),ones(1,200)],[ones(1,200)],[ones(1,200)],...
    [2*ones(1,200),ones(1,200)],[2*ones(1,200)],...
    [zeros(1,200)],[2*ones(1,200),2*ones(1,200)],...
    [1*ones(1,200)],[2*ones(1,200)],[2*ones(1,200)],[zeros(1,200)],...
    [2*ones(1,200),0*ones(1,200)],[0*ones(1,200)],[1*ones(1,200),0*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[1*ones(1,200)],[2*ones(1,200)],[2*ones(1,200),1*ones(1,200)]...
    };      % area manual annotation - before third main block - (0=V1 fentanest, 1=LM fentanest, 2=RL fentanest, 3=AL fentanest)

% layer attribution from histology - fist block - first shank
listSessions(10,:) = {...
    [4*ones(size(1:7)),2*ones(size(8:32)),],...        %'23_11_2016' b3 s1  Channels 1-4 are in layer 4; channelS 5-7 are between layers 4 and 3; channels 8-32 are in layer 3.
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [5*ones(size(1:11)),4*ones(size(12:20)),2*ones(size(21:32))],...            % '17_03_2017' b5 s1  Channels 1-8 are in layer 5; channels 9-11 are between layers 5 and 4; channels 12-18 are in layer 4; channels 19-20 are between layers 4 and 3; channels 21-32 are in layers 2/3. (In questa istologia non riesco a separare i layers 2 e 3)
    [5*ones(size(1:22)),4*ones(size(23:29)),2*ones(size(30:32))],...            % '21_03_2017' b2 s1  Channels 1-21 are in layer 5; channel 22 is between layers 5 and 4; channels 23-28 are in layer 4; channel 29 is between layers 4 and 3; channels 30-32 are in layer 3. 
    [5*ones(size(1:23)),4*ones(size(24:32))],...                                % '22_03_2017' b2 s1  Channels 1-22 are in layer 5; channel 23 is between layers 5 and 4; channels 24-31 are in layer 4; channel 32 is between layers 4 and 3. MODIFICATO
    [5*ones(size(1:9)),4*ones(size(10:15)),2*ones(size(16:28)),1*ones(size(29:32))],...              % '27_03_2017' b3 s1  Channels 1-8 are in layer 5; channel 9 is between layers 5 and 4; channels 10-15 are in layer 4; 16-28 in layers2/3 e da 28-32 in layer1.
    [5*ones(size(1:8)),4*ones(size(9:23)),2*ones(size(24:32))],...              % '04_05_2017' b2 s1  Channels 1-7 are in layer 5; channel 8 is between layers 5 and 4; channels 9-21 are in layer 4; channels 22-24 are between layers 4 and 3; channels 25-32 in layer 2/3.
    [5*ones(size(1:9)),4*ones(size(10:15)),2*ones(size(16:29)),1*ones(size(30:32))],...              % '09_05_2017' b2 s1  Channels 1-8 are in layer 5; channel 9 is between layers 5 and 4; channels 10-14 are in layer 4; channel 15 is between layers 4 and 3; channels 16-32 are in layer 2/3. 
    [0*ones(size(1:32))],...    % 10 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [5*ones(size(1:6)),4*ones(size(7:11)),2*ones(size(12:22)),1*ones(size(22:32))],...  % '26_09_2017' b8 s1 Channels 1-3 are in layer 5; channel 4 is between layers 5 and 4; channels 5-9 are in layer 4; channel 10 is between layers 4 and 3; channels 11-18 are in layer 2/3, channels 19-20 are between layers 2 and 1; channels 21-24 are in layer 1; channels 25-32 are out of the slice.
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...    % 20 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))]...    % 30 ---------------------
    };
% layer attribution from histology - second block - first shank
listSessions(11,:) = {...
    [4*ones(size(1:27)),2*ones(size(28:32))],...        % '23_11_2016'	b5 s1  Channels 1-27 are in layer 4; channels 28-31 are between layers 4 and 3; channel 32 is in layer 3.
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [5*ones(size(1:4)),4*ones(size(5:17)),2*ones(size(18:32))],...        % '17_03_2017' b8 s1 Channels 1-2 are in layer 5; channels 3-4 are between layers 5 and 4; channels 5-16 are in layer 4; channels 17-18 are between layers 4 and 3; channels 19-30 are in layer 3; channel 31 is in between layers 3 and 2; channel 32 is in layer 2. 
    [5*ones(size(1:19)),4*ones(size(20:27)),2*ones(size(28:32))],...      % '21_03_2017' b8 s1 Channels 1-18 are in layer 5; channel 19 is between layers 5 and 4; channels 20-26 are in layer 4; channel 27 is between layers 4 and 3; channels 28-32 are in layer 3.  
    [5*ones(size(1:12)),4*ones(size(13:23)),2*ones(size(24:32))],...      % '22_03_2017' b5 s1 Channels 1-11 are in layer 5; channels 12-13 are between layers 5 and 4; channels 14-21 are in layer 4; channel 22 is between layers 4 and 3; channels 23-32 are in layers 2/3. (In questa istologia non riesco a separare i layers 2 e 3) MODIFICATO
    [5*ones(size(1:16)),4*ones(size(17:22)),2*ones(size(23:29)),1*ones(size(30:32))],... % '27_03_2017'  b9 s1 Channels 1-16 are in layer 5; channels 17-22 are in layer 4, 23 al 29 in layer 2/3 e dal 30-32 in layer1.)
    [5*ones(size(1:20)),4*ones(size(21:30)),2*ones(size(31:32))],...      % '04_05_2017' b6 s1  Channels 1-18 are in layer 5; channels 19-20 are between layers 5 and 4; channels 21-30 are in layer 4; channels 31-32 are between layers 4 and 3.
    [5*ones(size(1:16)),4*ones(size(17:23)),2*ones(size(24:32))],...      % '09_05_2017' b6 s1  Channels 1-15 are in layer 5; channel 16 is between layers 5 and 4; channels 17-22 are in layer 4; channel 23 is between layers 4 and 3; channels 24-32 are in layers 2/3.
    [0*ones(size(1:32))],...    % 10 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...    % 20 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))]...    % 30 ---------------------
    };

% layer attribution from histology - fist block - second shank
listSessions(12,:) = {...
    [0*ones(size(1:32))],...        %'23_11_2016' b3 s2  Channels 1-8 are in layer 5; channel 9 is between layers 5 and 4; channels 10-14 are in layer 4; channel 15 is between layers 4 and 3; channels 16-26 are in layers 2/3 (NB: non riesco a separare i layers); channels 27-31 are in layer 1; channel 32 is out of the slice
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...    % 10 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [5*ones(size(1:9)),4*ones(size(10:15)),2*ones(size(16:27)),1*ones(size(28:32))],...   % '26_09_2017' b8 s2 Channels 1-8 are in layer 5; channel 9 is between layers 5 and 4; channels 10-14 are in layer 4; channel 15 is between layers 4 and 3; channels 16-26 are in layers 2/3; channels 27-31 are in layer 1; channel 32 is out of the slice.
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...    % 20 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))]...    % 30 ---------------------
    };
% layer attribution from histology - second shank
listSessions(13,:) = {...
    [0*ones(size(1:32))],...        %'23_11_2016'	b5 s2
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...    % 10 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...    % 20 ---------------------
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))],...
    [0*ones(size(1:32))]...    % 30 ---------------------
    };

pars.listSessions = listSessions;

%% stimulus parameters

pars.stimPars = [];

pars.stimPars.SF=[0.02,0.04];                        % (Spatial Frequencies)
pars.stimPars.TF=[2,6];                              % (Temporal Frequencies)
pars.stimPars.DIR=0:30:330;                          % (DIRections)
pars.stimPars.numSF=length(pars.stimPars.SF);
pars.stimPars.numTF=length(pars.stimPars.TF);
pars.stimPars.numDIR=length(pars.stimPars.DIR);
pars.stimPars.noiselength=1818;

pars.stimPars.stimulustype={'grating','plaid'};
pars.stimPars.numstimulustype=length(pars.stimPars.stimulustype);

pars.stimPars.gratings_bitcodes=[100:111,112:123,124:135,136:147];   % (first increasing SF, second increasing TF, DIR increasing within chunk)
pars.stimPars.plaids_bitcodes=[350:361,362:373,374:385,386:397];     % (first increasing SF, second increasing TF, DIR increasing within chunk)
pars.stimPars.flashes_bitcodes=[554,555];                            % (first black, second white)
pars.stimPars.noises_bitcodes=[1:40];                                % (30 s = 909 frames chunks of gaussianly correlated 0.04 cpd-matched 5s contrast modulated noise)

pars.stimPars.ntrials=20;
pars.stimPars.winwidth=0.85;
pars.stimPars.pre_time=0.8;
pars.stimPars.post_time=0.8;
pars.stimPars.stim_time=0.9;
pars.stimPars.pre_delay=0.05;
pars.stimPars.post_delay=0.1;
pars.stimPars.frame_duration=0.033;
pars.stimPars.num_frames=27;

pars.neuPars.ctypes={'good','mua','noise'};
pars.neuPars.clabels={2,1,0};

%% spike triggered average parameters

pars.STA_depth=10;
pars.STA_width=32;
pars.STA_height=18;
pars.STA_ridgeparam=3;
pars.interp_factor=20;
pars.crop_pixel_size=250;% renamed from crop_factor = 2.25; ----> corresponding to ma=cropfactor*2.35*(sigma) = 317.25 
pars.contrast_threshold=8.5;

%% LNP prediction parameters

pars.prediction_isi=25;

%% various pars

pars.wave_samp_freq=2.44140625e+04;
pars.LFP_samp_freq=6.1035e+02;
pars.shanklength=800;

%% manually annotated broken channels list (LFP)

% % 0
% '23_11_2016'    [3,5]
% '02_03_2017'    [2,6]
% '16_03_2017'    [4,7]
% '17_03_2017'    [5,9]
% '21_03_2017'    [2,8]
% '22_03_2017'    [2,5]
% '27_03_2017'    [3,9]
% '04_05_2017'    [2,6]
% '09_05_2017'    [2,6]
% '16_05_2017'    [7]
% % 10
% '04_09_2017'    [5,10]
% '12_09_2017'    [3,6]   2shank!
% '14_09_2017'    [7]
% '18_09_2017'    [4,8]
% '20_09_2017'    [3]
% '26_09_2017'    [8]
% '29_09_2017'    [4,6]
% '08_11_2017'    [4]
% '13_11_2017'    [4]
% '28_11_2017'    [5]
% % 20
% '19_12_2017'    [7]
% '13_02_2018'    [6,10]
% '20_02_2018'    [5]
% '06_03_2018'    [5,8]
% '15_03_2018'    [3]
% '22_03_2018'    [3]
% '12_04_2018'    [6]
% '19_04_2018'    [4]
% '03_05_2018'    [3]
% '09_05_2018'    [4,11]
% % 30

% ---------------------------------------------------------------------
%'23_11_2016'	[3,5]
pars.broken_channels{1}{1}{1}=[3,7,11,18,26];
pars.broken_channels{1}{2}{1}=[3,7,11,18,26];
%'02_03_2017'   [2,6]
pars.broken_channels{2}{1}{1}=[];
pars.broken_channels{2}{2}{1}=[];
%'16_03_2017'   [4,7]
pars.broken_channels{3}{1}{1}=[3,7];
pars.broken_channels{3}{2}{1}=[3,7];
%'17_03_2017'   [5,9]
pars.broken_channels{4}{1}{1}=[];
pars.broken_channels{4}{2}{1}=[];
%'21_03_2017'   [2,8]
pars.broken_channels{5}{1}{1}=[3,7,11,18,26];
pars.broken_channels{5}{2}{1}=[3,7,11,18,26];
%'22_03_2017'   [2,5]
pars.broken_channels{6}{1}{1}=[3,7,11,18,26];
pars.broken_channels{6}{2}{1}=[];
%'27_03_2017'   [3,9]
pars.broken_channels{7}{1}{1}=[7];
pars.broken_channels{7}{2}{1}=[19,24];
%'04_05_2017'   [2,6]
pars.broken_channels{8}{1}{1}=[19,24];
pars.broken_channels{8}{2}{1}=[12,19,24];
%'09_05_2017'   [2,6]
pars.broken_channels{9}{1}{1}=[3,7];
pars.broken_channels{9}{2}{1}=[];
%'16_05_2017'   [7]
pars.broken_channels{10}{1}{1}=[3,7,11];
% 10 ------------------------------------
%'04_09_2017'    [5,10]
pars.broken_channels{11}{1}{1}=[];
pars.broken_channels{11}{2}{1}=[];
%'12_09_2017'    [3,6]
pars.broken_channels{12}{1}{1}=[];
pars.broken_channels{12}{1}{2}=[];
pars.broken_channels{12}{2}{1}=[];
pars.broken_channels{12}{2}{2}=[];
%'14_09_2017'    [7]
pars.broken_channels{13}{1}{1}=[];
pars.broken_channels{13}{1}{2}=[];
%'18_09_2017'    [4,8]
pars.broken_channels{14}{1}{1}=[];
pars.broken_channels{14}{1}{2}=[];
pars.broken_channels{14}{2}{1}=[];
pars.broken_channels{14}{2}{2}=[];
%'20_09_2017'    [3]
pars.broken_channels{15}{1}{1}=[];
pars.broken_channels{15}{1}{2}=[];
%'26_09_2017'    [8]
pars.broken_channels{16}{1}{1}=[];
pars.broken_channels{16}{1}{2}=[];
%'29_09_2017'    [4,6]
pars.broken_channels{17}{1}{1}=[];
pars.broken_channels{17}{1}{2}=[];
pars.broken_channels{17}{2}{1}=[];
pars.broken_channels{17}{2}{2}=[];
%'08_11_2017'    [4]
pars.broken_channels{18}{1}{1}=[];
pars.broken_channels{18}{1}{2}=[20];
%'13_11_2017'    [4]
pars.broken_channels{19}{1}{1}=[];
pars.broken_channels{19}{1}{2}=[20,26];
%'28_11_2017'    [5]
pars.broken_channels{20}{1}{1}=[];
pars.broken_channels{20}{1}{2}=[];
% 20 ------------------------------------
%'19_12_2017'    [7]
pars.broken_channels{21}{1}{1}=[];
pars.broken_channels{21}{1}{2}=[];
%'13_02_2018'    [6,10]
pars.broken_channels{22}{1}{1}=[];
pars.broken_channels{22}{1}{2}=[];
pars.broken_channels{22}{2}{1}=[];
pars.broken_channels{22}{2}{2}=[];
%'20_02_2018'    [5]
pars.broken_channels{23}{1}{1}=[];
pars.broken_channels{23}{1}{2}=[];
%'06_03_2018'    [5,8]
pars.broken_channels{24}{1}{1}=[];
pars.broken_channels{24}{1}{2}=[];
pars.broken_channels{24}{2}{1}=[];
pars.broken_channels{24}{2}{2}=[];
%'15_03_2018'    [3]
pars.broken_channels{25}{1}{1}=[];
pars.broken_channels{25}{1}{2}=[];
%'22_03_2018'    [3]
pars.broken_channels{26}{1}{1}=[];
pars.broken_channels{26}{1}{2}=[];
%'12_04_2018'    [6]
pars.broken_channels{27}{1}{1}=[];
pars.broken_channels{27}{1}{2}=[];
%'19_04_2018'    [4]
pars.broken_channels{28}{1}{1}=[];
pars.broken_channels{28}{1}{2}=[];
%'03_05_2018'    [3]
pars.broken_channels{29}{1}{1}=[];
pars.broken_channels{29}{1}{2}=[];
%'09_05_2018'    [4,11]
pars.broken_channels{30}{1}{1}=[];
pars.broken_channels{30}{1}{2}=[7,9,11,13,15,17,19,21,22,29];
pars.broken_channels{30}{2}{1}=[];
pars.broken_channels{30}{2}{2}=[];
% 30 ------------------------------------

% ---------------------------------------------------------------------

%% manually annotated channel depth list (LFP)

% ---------------------------------------------------------------------
% '23_11_2016'    [3,5]
pars.channels_depth_with{1}{1}{1}=linspace(514,324,32);
pars.channels_depth_without{1}{1}{1}=linspace(524,361,32);
pars.channels_bregma{1}{1}{1}=-4.84;
pars.channels_depth_with{1}{2}{1}=linspace(500,480,32);
pars.channels_depth_without{1}{2}{1}=linspace(500,480,32);
pars.channels_bregma{1}{2}{1}=-4.84;
% '02_03_2017'    [2,6]
pars.channels_depth_with{2}{1}{1}=NaN(1,32);
pars.channels_depth_without{2}{1}{1}=NaN(1,32);
pars.channels_bregma{2}{1}{1}=NaN;
pars.channels_depth_with{2}{2}{1}=NaN(1,32);
pars.channels_depth_without{2}{2}{1}=NaN(1,32);
pars.channels_bregma{2}{2}{1}=NaN;
% '16_03_2017'    [4,7]
pars.channels_depth_with{3}{1}{1}=NaN(1,32);
pars.channels_depth_without{3}{1}{1}=NaN(1,32);
pars.channels_bregma{3}{1}{1}=NaN;
pars.channels_depth_with{3}{2}{1}=NaN(1,32);
pars.channels_depth_without{3}{2}{1}=NaN(1,32);
pars.channels_bregma{3}{2}{1}=NaN;
% '17_03_2017'    [5,9]
pars.channels_depth_with{4}{1}{1}=linspace(817,332,32);
pars.channels_depth_without{4}{1}{1}=linspace(817,332,32);
pars.channels_bregma{4}{1}{1}=-6.71;
pars.channels_depth_with{4}{2}{1}=linspace(774,172,32);
pars.channels_depth_without{4}{2}{1}=linspace(774,172,32);
pars.channels_bregma{4}{2}{1}=-6.74;
% '21_03_2017'    [2,8]
pars.channels_depth_with{5}{1}{1}=linspace(1112,413,32);
pars.channels_depth_without{5}{1}{1}=linspace(1112,413,32);
pars.channels_bregma{5}{1}{1}=-5.65;
pars.channels_depth_with{5}{2}{1}=linspace(981+50,287+50,32);              % modifica 07_08_2018
pars.channels_depth_without{5}{2}{1}=linspace(981+50,287+50,32);           % modifica 07_08_2018
pars.channels_bregma{5}{2}{1}=-5.29;
% '22_03_2017'    [2,5]
pars.channels_depth_with{6}{1}{1}=linspace(1119,539,32);
pars.channels_depth_without{6}{1}{1}=linspace(1119,539,32);
pars.channels_bregma{6}{1}{1}=-6.21;
pars.channels_depth_with{6}{2}{1}=linspace(965,296,32);
pars.channels_depth_without{6}{2}{1}=linspace(965,296,32);
pars.channels_bregma{6}{2}{1}=-6.21;
% '27_03_2017'    [3,9]
pars.channels_depth_with{7}{1}{1}=linspace(638,0,32);
pars.channels_depth_without{7}{1}{1}=linspace(638,0,32);
pars.channels_bregma{7}{1}{1}=-4.74;
pars.channels_depth_with{7}{2}{1}=linspace(893,120,32);
pars.channels_depth_without{7}{2}{1}=linspace(893,120,32);
pars.channels_bregma{7}{2}{1}=-4.98;
% '04_05_2017'    [2,6]
pars.channels_depth_with{8}{1}{1}=linspace(758,209,32);
pars.channels_depth_without{8}{1}{1}=linspace(758,209,32);
pars.channels_bregma{8}{1}{1}=-6.54;
pars.channels_depth_with{8}{2}{1}=linspace(778+150,201+150,32);                     % modifica 07_08_2018
pars.channels_depth_without{8}{2}{1}=linspace(778+150,201+150,32);                  % modifica 07_08_2018
pars.channels_bregma{8}{2}{1}=-6.81;
% '09_05_2017'    [2,6]  % non ho neanche scritto quelle con funghetto
pars.channels_depth_with{9}{1}{1}=linspace(775,0,32);
pars.channels_depth_without{9}{1}{1}=linspace(775,0,32);
pars.channels_bregma{9}{1}{1}=-5.23;
pars.channels_depth_with{9}{2}{1}=linspace(828,188,32);
pars.channels_depth_without{9}{2}{1}=linspace(828,188,32);
pars.channels_bregma{9}{2}{1}=-5.68;
% '16_05_2017'    [7]
pars.channels_depth_with{10}{1}{1}=NaN(1,32);
pars.channels_depth_without{10}{1}{1}=NaN(1,32);
pars.channels_bregma{10}{1}{1}=NaN;
% % 10
% '04_09_2017'    [5,10]
pars.channels_depth_with{11}{1}{1}=NaN(1,32);
pars.channels_depth_without{11}{1}{1}=NaN(1,32);
pars.channels_bregma{11}{1}{1}=NaN;
pars.channels_depth_with{11}{2}{1}=NaN(1,32);
pars.channels_depth_without{11}{2}{1}=NaN(1,32);
pars.channels_bregma{11}{2}{1}=NaN;
% '12_09_2017'    [3,6]   2shank!
pars.channels_depth_with{12}{1}{1}=NaN(1,32);
pars.channels_depth_without{12}{1}{1}=NaN(1,32);
pars.channels_bregma{12}{1}{1}=NaN;
pars.channels_depth_with{12}{1}{2}=NaN(1,32);
pars.channels_depth_without{12}{1}{2}=NaN(1,32);
pars.channels_bregma{12}{1}{2}=NaN;
pars.channels_depth_with{12}{2}{1}=NaN(1,32);
pars.channels_depth_without{12}{2}{1}=NaN(1,32);
pars.channels_bregma{12}{2}{1}=NaN;
pars.channels_depth_with{12}{2}{2}=NaN(1,32);
pars.channels_depth_without{12}{2}{2}=NaN(1,32);
pars.channels_bregma{12}{2}{2}=NaN;
% '14_09_2017'    [7]
pars.channels_depth_with{13}{1}{1}=NaN(1,32);
pars.channels_depth_without{13}{1}{1}=NaN(1,32);
pars.channels_bregma{13}{1}{1}=NaN;
pars.channels_depth_with{13}{1}{2}=NaN(1,32);
pars.channels_depth_without{13}{1}{2}=NaN(1,32);
pars.channels_bregma{13}{1}{2}=NaN;
% '18_09_2017'    [4,8]
pars.channels_depth_with{14}{1}{1}=NaN(1,32);
pars.channels_depth_without{14}{1}{1}=NaN(1,32);
pars.channels_bregma{14}{1}{1}=NaN;
pars.channels_depth_with{14}{1}{2}=NaN(1,32);
pars.channels_depth_without{14}{1}{2}=NaN(1,32);
pars.channels_bregma{14}{1}{2}=NaN;
pars.channels_depth_with{14}{2}{1}=NaN(1,32);
pars.channels_depth_without{14}{2}{1}=NaN(1,32);
pars.channels_bregma{14}{2}{1}=NaN;
pars.channels_depth_with{14}{2}{2}=NaN(1,32);
pars.channels_depth_without{14}{2}{2}=NaN(1,32);
pars.channels_bregma{14}{2}{2}=NaN;
% '20_09_2017'    [3]
pars.channels_depth_with{15}{1}{1}=NaN(1,32);
pars.channels_depth_without{15}{1}{1}=NaN(1,32);
pars.channels_bregma{15}{1}{1}=NaN;
pars.channels_depth_with{15}{1}{2}=NaN(1,32);
pars.channels_depth_without{15}{1}{2}=NaN(1,32);
pars.channels_bregma{15}{1}{2}=NaN;
% '26_09_2017'    [8]
pars.channels_depth_with{16}{1}{1}=linspace(700,0,32);
pars.channels_depth_without{16}{1}{1}=linspace(700,0,32);
pars.channels_bregma{16}{1}{1}=-6.89;
pars.channels_depth_with{16}{1}{2}=linspace(769,0,32);
pars.channels_depth_without{16}{1}{2}=linspace(769,0,32);
pars.channels_bregma{16}{1}{2}=-6.89;
% '29_09_2017'    [4,6]
pars.channels_depth_with{17}{1}{1}=NaN(1,32);
pars.channels_depth_without{17}{1}{1}=NaN(1,32);
pars.channels_bregma{17}{1}{1}=NaN;
pars.channels_depth_with{17}{1}{2}=NaN(1,32);
pars.channels_depth_without{17}{1}{2}=NaN(1,32);
pars.channels_bregma{17}{1}{2}=NaN;
pars.channels_depth_with{17}{2}{1}=NaN(1,32);
pars.channels_depth_without{17}{2}{1}=NaN(1,32);
pars.channels_bregma{17}{2}{1}=NaN;
pars.channels_depth_with{17}{2}{2}=NaN(1,32);
pars.channels_depth_without{17}{2}{2}=NaN(1,32);
pars.channels_bregma{17}{2}{2}=NaN;
% '08_11_2017'    [4]
pars.channels_depth_with{18}{1}{1}=NaN(1,32);
pars.channels_depth_without{18}{1}{1}=NaN(1,32);
pars.channels_bregma{18}{1}{1}=NaN;
pars.channels_depth_with{18}{1}{2}=NaN(1,32);
pars.channels_depth_without{18}{1}{2}=NaN(1,32);
pars.channels_bregma{18}{1}{2}=NaN;
% '13_11_2017'    [4]
pars.channels_depth_with{19}{1}{1}=NaN(1,32);
pars.channels_depth_without{19}{1}{1}=NaN(1,32);
pars.channels_bregma{19}{1}{1}=NaN;
pars.channels_depth_with{19}{1}{2}=NaN(1,32);
pars.channels_depth_without{19}{1}{2}=NaN(1,32);
pars.channels_bregma{19}{1}{2}=NaN;
% '28_11_2017'    [5]
pars.channels_depth_with{20}{1}{1}=NaN(1,32);
pars.channels_depth_without{20}{1}{1}=NaN(1,32);
pars.channels_bregma{20}{1}{1}=NaN;
pars.channels_depth_with{20}{1}{2}=NaN(1,32);
pars.channels_depth_without{20}{1}{2}=NaN(1,32);
pars.channels_bregma{20}{1}{2}=NaN;
% % 20
% '19_12_2017'    [7]
pars.channels_depth_with{21}{1}{1}=NaN(1,32);
pars.channels_depth_without{21}{1}{1}=NaN(1,32);
pars.channels_bregma{21}{1}{1}=NaN;
pars.channels_depth_with{21}{1}{2}=NaN(1,32);
pars.channels_depth_without{21}{1}{2}=NaN(1,32);
pars.channels_bregma{21}{1}{2}=NaN;
% '13_02_2018'    [6,10]
pars.channels_depth_with{22}{1}{1}=NaN(1,32);
pars.channels_depth_without{22}{1}{1}=NaN(1,32);
pars.channels_bregma{22}{1}{1}=NaN;
pars.channels_depth_with{22}{1}{2}=NaN(1,32);
pars.channels_depth_without{22}{1}{2}=NaN(1,32);
pars.channels_bregma{22}{1}{2}=NaN;
pars.channels_depth_with{22}{2}{1}=NaN(1,32);
pars.channels_depth_without{22}{2}{1}=NaN(1,32);
pars.channels_bregma{22}{2}{1}=NaN;
pars.channels_depth_with{22}{2}{2}=NaN(1,32);
pars.channels_depth_without{22}{2}{2}=NaN(1,32);
pars.channels_bregma{22}{2}{2}=NaN;
% '20_02_2018'    [5]
pars.channels_depth_with{23}{1}{1}=NaN(1,32);
pars.channels_depth_without{23}{1}{1}=NaN(1,32);
pars.channels_bregma{23}{1}{1}=NaN;
pars.channels_depth_with{23}{1}{2}=NaN(1,32);
pars.channels_depth_without{23}{1}{2}=NaN(1,32);
pars.channels_bregma{23}{1}{2}=NaN;
% '06_03_2018'    [5,8]
pars.channels_depth_with{24}{1}{1}=NaN(1,32);
pars.channels_depth_without{24}{1}{1}=NaN(1,32);
pars.channels_bregma{24}{1}{1}=NaN;
pars.channels_depth_with{24}{1}{2}=NaN(1,32);
pars.channels_depth_without{24}{1}{2}=NaN(1,32);
pars.channels_bregma{24}{1}{2}=NaN;
pars.channels_depth_with{24}{2}{1}=NaN(1,32);
pars.channels_depth_without{24}{2}{1}=NaN(1,32);
pars.channels_bregma{24}{2}{1}=NaN;
pars.channels_depth_with{24}{2}{2}=NaN(1,32);
pars.channels_depth_without{24}{2}{2}=NaN(1,32);
pars.channels_bregma{24}{2}{2}=NaN;
% '15_03_2018'    [3]
pars.channels_depth_with{25}{1}{1}=NaN(1,32);
pars.channels_depth_without{25}{1}{1}=NaN(1,32);
pars.channels_bregma{25}{1}{1}=NaN;
pars.channels_depth_with{25}{1}{2}=NaN(1,32);
pars.channels_depth_without{25}{1}{2}=NaN(1,32);
pars.channels_bregma{25}{1}{2}=NaN;
% '22_03_2018'    [3]
pars.channels_depth_with{26}{1}{1}=NaN(1,32);
pars.channels_depth_without{26}{1}{1}=NaN(1,32);
pars.channels_bregma{26}{1}{1}=NaN;
pars.channels_depth_with{26}{1}{2}=NaN(1,32);
pars.channels_depth_without{26}{1}{2}=NaN(1,32);
pars.channels_bregma{26}{1}{2}=NaN;
% '12_04_2018'    [6]
pars.channels_depth_with{27}{1}{1}=NaN(1,32);
pars.channels_depth_without{27}{1}{1}=NaN(1,32);
pars.channels_bregma{27}{1}{1}=NaN;
pars.channels_depth_with{27}{1}{2}=NaN(1,32);
pars.channels_depth_without{27}{1}{2}=NaN(1,32);
pars.channels_bregma{27}{1}{2}=NaN;
% '19_04_2018'    [4]
pars.channels_depth_with{28}{1}{1}=NaN(1,32);
pars.channels_depth_without{28}{1}{1}=NaN(1,32);
pars.channels_bregma{28}{1}{1}=NaN;
pars.channels_depth_with{28}{1}{2}=NaN(1,32);
pars.channels_depth_without{28}{1}{2}=NaN(1,32);
pars.channels_bregma{28}{1}{2}=NaN;
% '03_05_2018'    [3]
pars.channels_depth_with{29}{1}{1}=NaN(1,32);
pars.channels_depth_without{29}{1}{1}=NaN(1,32);
pars.channels_bregma{29}{1}{1}=NaN;
pars.channels_depth_with{29}{1}{2}=NaN(1,32);
pars.channels_depth_without{29}{1}{2}=NaN(1,32);
pars.channels_bregma{29}{1}{2}=NaN;
% '09_05_2018'    [4,11]
pars.channels_depth_with{30}{1}{1}=NaN(1,32);
pars.channels_depth_without{30}{1}{1}=NaN(1,32);
pars.channels_bregma{30}{1}{1}=NaN;
pars.channels_depth_with{30}{1}{2}=NaN(1,32);
pars.channels_depth_without{30}{1}{2}=NaN(1,32);
pars.channels_bregma{30}{1}{2}=NaN;
pars.channels_depth_with{30}{2}{1}=NaN(1,32);
pars.channels_depth_without{30}{2}{1}=NaN(1,32);
pars.channels_bregma{30}{2}{1}=NaN;
pars.channels_depth_with{30}{2}{2}=NaN(1,32);
pars.channels_depth_without{30}{2}{2}=NaN(1,32);
pars.channels_bregma{30}{2}{2}=NaN;

% ---------------------------------------------------------------------

end