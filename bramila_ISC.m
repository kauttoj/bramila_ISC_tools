function results=bramila_ISC(cfg)
% results = bramila_ISC(cfg)
%
% Compute ISC for 4D braindata. Input 4D nifti files (one per subject/group).
% ISC is based on one-vs-others approach omitting all pair-wise comparisons. This should improve SNR
% of ISC method as we are using "correlation of averages" instead of "averages of corralations" approach.
% Statistics are based on permutations that remove (presumed) temporal syncrony. 
% Resulting 3D statistics include standard FDR/FWE, TFCE and cluster extent.
%
% The code & idea was inspired by Brainiak Python library (U. Hasson Lab)
%
% Note 1: By default, all computations are performed using cluster jobs (SLURM). You can execute this function, e.g., from a head node (no heavy computing).
% Note 2: This function requires "bramila_ISC_worker.m"
%
% Input parameters:
%
%   cgf.infiles= paths to NIFTIs (MANDATORY)
%   cfg.mask = 3D brain mask or path to NIFTI, all values >0 are included (MANDATORY)
%   cfg.group_id = vector of group membership for each file, only values 1,2 and 0 are allowed. Must match total number of subjects! (MANDATORY)
%   cfg.workfolder = working directory where all tempfiles and results go (MANDATORY)
%   cfg.parallel_jobs = number of jobs to compute permutations. If 0, only single core is used (OPTIONAL)
%   cfg.iter = number of permutations/job (default 20) (OPTIONAL)
%   cfg.overwrite = do we overwrite existing files (0=none,1=only results,2=results and data), default = 0 (OPTIONAL)
%   cfg.niftitools_path = path to "load_nii" function, if not found, must set manually (OPTIONAL)
%   cfg.memory_limit_GB = rought limit for allowed worker memory, data is splitted if needed (default 40GB, OPTIONAL)
%   cfg.doLocalSerial = process all locally ONLY FOR TESTING PURPOSES (OPTIONAL)
%
%  Note: Permutations are SLOW (~5-10min for one permutation per cpu), cfg.parallel_jobs ~ 100 highly recommended. 
%        Total permutation count is cfg.parallel_jobs*cfg.iter and should be >1000
%
% Output results struct with following fields:
%
%                      results.raw_tval_map: 3D voxel-wise t-values
%                      results.raw_tfce_map: 3D voxel-wise TFCE values
%                              results.mask: 3D mask
%                          results.mask_ind: mask indices
%                        results.cfg_struct: copy of your cfg
%                           results.stats.*: all statistical maps (see following)
%         results.stats.tfce_pval_corrected: TFCE p-vals FWER (maxstat) corrected
%       results.stats.tfce_pval_uncorrected: TFCE p-vals without correction
%        results.stats.raw_pval_uncorrected: uncorrected p-vals
%                results.stats.raw_pval_FDR: FDR corrected p-vals
%                results.stats.raw_pval_FWE: FWE corrected p-vals (maximum statistics)
%      results.stats.cluster_pval_corrected: cluster extent corrected p-vals (repeated over cluster)
%    results.stats.cluster_voxelwise_extent: number of voxels in the corresponding cluster
%          results.stats.cluster_thresholds: list of voxels extends for p<{0.05,0.01,0.001} used in correction 
%
%
% 4.1.2019 (ver 1.1), Janne Kauttonen, Aalto University
%
%   Example usage: 
%    cfg=[];
%
%    k=0;
%    k=k+1;cfg.infiles{k}='mynifti1.nii';
%    k=k+1;cfg.infiles{k}='mynifti2.nii';
%    k=k+1;cfg.infiles{k}='mynifti3.nii';
%    k=k+1;cfg.infiles{k}='mynifti4.nii';
%    k=k+1;cfg.infiles{k}='mynifti5.nii';
%    k=k+1;cfg.infiles{k}='mynifti6.nii';
%    k=k+1;cfg.infiles{k}='mynifti7.nii';
%    k=k+1;cfg.infiles{k}='mynifti8.nii';
%    k=k+1;cfg.infiles{k}='mynifti9.nii';
%    k=k+1;cfg.infiles{k}='mynifti10.nii';
%    k=k+1;cfg.infiles{k}='mynifti11.nii';
%    k=k+1;cfg.infiles{k}='mynifti12.nii';
%    k=k+1;cfg.infiles{k}='mynifti13.nii';
% 
%    cfg.group_id = zeros(1,13);
%    cfg.group_id(1:7)=1;
%    cfg.group_id(8:13])=2;
%    cfg.iter=10;
%    cfg.parallel_jobs = 100;
%    cfg.mask = 'mymaskfile.nii';    
%    cfg.workfolder = '/my/results/folder/'; % where to store tempdata and results
%
%    results = bramila_ISC(cfg);
%

% find path to worker code
worker_path = which('bramila_ISC_worker.m');
if isempty(worker_path)
   addpath(cfg.worker_path);
   assert(~isempty(which('load_nii.m')),'niftitools not found!');
else
   cfg.worker_path = fileparts(worker_path);
end

% niftitools path
niftitools_path = which('load_nii.m');
if isempty(niftitools_path )
   addpath(cfg.niftitools_path);
   assert(~isempty(which('load_nii.m')),'niftitools not found!');
else
   cfg.niftitools_path = fileparts(niftitools_path);
end

if(~isfield(cfg,'perm_method'))
   cfg.perm_method = 'circshift';
end
assert(ismember(cfg.perm_method,{'circshift'}),'permutation method must be ''circshift'' (no others implemented yet)!');

% use local machine (for testing)
if(~isfield(cfg,'doLocalSerial'))
   cfg.doLocalSerial = 0;
end

if(~isfield(cfg,'workfolder'))
    cfg.workfolder = tempdir();
    warning('Working folder not set, using system temp %s',cfg.workfolder);
else
    if ~exist(cfg.workfolder,'dir')
        mkdir(cfg.workfolder);
    end
end
assert(exist(cfg.workfolder,'dir')>0,'work folder does not exist (or failed to create one)!');

diaryfile = [cfg.workfolder,filesep,'bramila_ISC_processing_diary.txt'];
if exist(diaryfile,'file')
    delete(diaryfile);
end
diary(diaryfile)

fprintf('---- Running "bramila_ISC" to compute one-vs-others ISC statistics (%s) ----\n',datestr(now,'HH:MM:SS'));

% initialize job count
if(~isfield(cfg,'parallel_jobs'))
    cfg.parallel_jobs=100;
end

% how many parts to process data
if(~isfield(cfg,'data_split_parts'))
    cfg.data_split_parts=1;
end

% set memory limit
if(~isfield(cfg,'memory_limit_GB'))
    cfg.memory_limit_GB=40;
end
%assert(cfg.memory_limit_GB>=5,'You should have at least 5GB of memory!');

% p-val threshold for cluster extend correction
if ~isfield(cfg,'p_val_threshold')
   cfg.p_val_threshold = 0.01;
end
assert(cfg.p_val_threshold<=0.01,'cluster-forming threshold should be p<=0.01!'); % do not allow weaker thresholds

if ~isfield(cfg,'doFisherTransform')
   cfg.doFisherTransform = 1;
end

% initialize iter
if(~isfield(cfg,'iter'))
    cfg.iter=15;
end
assert(cfg.iter<=10000); % lets not go crazy!

if cfg.iter*cfg.parallel_jobs<1000
   warning('!! Only %i total iterations used, results are unreliable !!',cfg.iter*cfg.parallel_jobs);
end 

% initialize iter
if(~isfield(cfg,'group_id'))
    cfg.group_id=ones(1,length(cfg.infiles)); % all in one group
end

% group indices of rows & columns
group1_ind = find(cfg.group_id==1);
group2_ind = find(cfg.group_id==2);
group1_ind=group1_ind(:)';
group2_ind=group2_ind(:)';
Nsubj1 = length(group1_ind);
Nsubj2 = length(group2_ind);
Nsubs = Nsubj1+Nsubj2;
cfg.Nsubs=Nsubs;
cfg.infiles = cfg.infiles([group1_ind,group2_ind]);
group1_ind = 1:Nsubj1;
group2_ind =Nsubj1 + (1:Nsubj2);

cfg.group1_ind = group1_ind;
cfg.group2_ind = group2_ind;
cfg.Nsubj1=Nsubj1;
cfg.Nsubj2=Nsubj2;

% now all data is ordered by groups and only contain requested data

if not( (Nsubj2>2 && Nsubj1>2) || (Nsubj2==0 && Nsubj1>2))
    error('Cannot continue with only %i+%i subjects!',Nsubj1,Nsubj2);
end
assert(length(cfg.infiles) == Nsubs);

if Nsubj2==0
    fprintf('\nOne group with %i subjects, doing one-sample t-test (positive tail onlyl)\n',Nsubj1);
else
    fprintf('\nTwo groups with %i + %i subjects, comparing groups with two-sample t-test (both tails)\n',Nsubj1,Nsubj2);
end

% tempdata overwrite
if(~isfield(cfg,'overwrite'))
    cfg.overwrite=0;
end

% load mask
if ischar(cfg.mask)
    sprintf('Loading maskfile %s\n',cfg.mask);
    nii=load_nii(cfg.mask);
    mask=nii.img;
    clear nii;
else
    mask = cfg.mask;    
end
mask = mask>0;
cfg.mask=mask;
mask_ind = find(mask);
sz_mask = size(mask);
cfg.mask_ind = mask_ind;
cfg.Nvoxel = length(mask_ind);
cfg.sz_mask = sz_mask;

% options for TFCE, these are the defaults for 3D fMRI data
tfce_opts=[];
tfce_opts.tfce.H              = 2;                  % TFCE H parameter
tfce_opts.tfce.E              = 0.5;                % TFCE E parameter
tfce_opts.tfce.conn           = 26;                 % TFCE connectivity neighbourhood (26 all neighbors in 3D)
tfce_opts.tfce.deltah         = 0;             % Delta-h of the TFCE equation (0 = auto with 100 steps)
cfg.tfce_opts = tfce_opts;

% make sure mask makes sense for a typical experiment
mask_coverage_prc = 100*nnz(mask)/numel(mask);
assert(mask_coverage_prc>5 && mask_coverage_prc<50,'Mask does not include full brain. Either fix your mask or disable this check.');
% check dimensions

infiles = [cfg.workfolder,filesep,sprintf('bramila_ISC_tempdata_filelist.txt')];

fprintf('\nStage 0: Creating datasets as voxels x timepoints (%s)\n',datestr(datetime('now')));

%% Stage 0: Create z-scored masked datasets of size voxels x timepoints

stage=0;
for i = 1:Nsubs
    file=cfg.infiles{i};
    group = 1+(i>Nsubj1);
    [~,f] = fileparts(file);    
    cfg.stage0_cfgfiles{i} = [cfg.workfolder,filesep,sprintf('stage0_cfg_sub%i.mat',i)];
    cfg.stage0_jobfiles{i} = [cfg.workfolder,filesep,sprintf('stage0_jobfile_sub%i',i)];
    cfg.stage0_resultfiles{i} = [cfg.workfolder,filesep,sprintf('%s_bramila_ISC_tempdata_sub%i_group%i.mat',f,i,group)];  
    save(cfg.stage0_cfgfiles{i},'cfg','stage','-v7.3');        
end
for i = 1:Nsubs
    cfg.process_index = i;
    save(cfg.stage0_cfgfiles{i},'cfg','-append');
end
cfg.process_index = [];

tic();
jobnames = cell(1,Nsubs);
lognames = jobnames;
wait_for=true(1,Nsubs);
for i = 1:Nsubs
    if ~exist(cfg.stage0_resultfiles{i},'file') || cfg.overwrite>1
        % 15GB memory limit should be enough for any dataset
        [jobnames{i},lognames{i}] = sendjob(15000,cfg.stage0_jobfiles{i},cfg.stage0_cfgfiles{i},cfg.worker_path,'bramila_ISC_worker',cfg.doLocalSerial);
    else
        wait_for(i)=false;
        stage=1;
        save(cfg.stage0_cfgfiles{i},'stage','-append');
    end
end
fprintf('... %i jobs submitted (%s), waiting...\n',sum(wait_for),datestr(datetime('now')));
wait_for_jobs(cfg.stage0_cfgfiles(wait_for),jobnames(wait_for),cfg.stage0_jobfiles(wait_for),lognames(wait_for),1,true);

% OLD CODE FOR LOCAL DATA GENERATION:
%         fprintf('...loading and saving data for subject %i/%i (group %i): %s\n',k,Nsubs,group,file);
%         nii=load_nii(file);
%         data = nii.img;
%         for i=1:3
%             assert(sz_mask(i)==size(data,i));
%         end
%         data = reshape(data,[],size(data,4));
%         data = zscore(data(mask_ind,:),[],2);
%         data = single(data); % save as single to save space and memory
%         sz_data = size(data);
%         frames(k) = sz_data(2);
%         save(tempfiles{k},'data','file','mask','sz_data','-v7');
%         clear nii data

% check number of frames, mask and bad frames in data, cannot continue if there are inconsistencies
fprintf('Checking consistency of datasets (timepoints, mask and bad voxels)\n');
frames=nan(1,Nsubs);
bad_voxels = [];
fout = fopen(infiles,'w');
for i = 1:Nsubs
    group = 1+(i>Nsubj1);
    olddata = load(cfg.stage0_resultfiles{i},'sz_data','mask','bad_voxels'); % check mask
    frames(i) = olddata.sz_data(2);
    assert(nnz(mask ~= olddata.mask)==0,'data mask does not match current group mask!');    
    bad_voxels = union(bad_voxels,olddata.bad_voxels);    
    clear olddata
    fprintf(fout,'%s\t%i\t%i\n',cfg.stage0_resultfiles{i},group,frames(i));
end
fclose(fout);

% make sure there are no bad voxels
if ~isempty(bad_voxels)>0
   error('Total %i bad voxels found over subjects! Either fix your data or reduce mask size!',length(bad_voxels));
end

% make sure frame count match
assert(length(unique(frames))==1,'Number of timepoints does not match between subjects!');

cfg.Ntime = frames(1);

% how much memory to keep all subjects in memory
max_elements = max(frames)*length(mask_ind);
mem_estimate_MB = 1500 + 4*max_elements*(Nsubs+2)/1e+6; % 4 bytes per variable for a single, 1.5GB just to load Matlab

% how many parts?
cfg.data_split_parts = ceil((mem_estimate_MB/1000)/cfg.memory_limit_GB);

if cfg.data_split_parts>1
	mem_estimate_MB = 1.1*(4*max_elements/1e+6 + 1500 + (4*max_elements*(Nsubs+2)/cfg.data_split_parts)/1e+6);
    cfg.data_split_parts = cfg.data_split_parts + double((mem_estimate_MB/1000-cfg.memory_limit_GB)>0); % refine first estimate
	mem_estimate_MB = 1.1*(4*max_elements/1e+6 + 1500 + (4*max_elements*(Nsubs+2)/cfg.data_split_parts)/1e+6); % recompute
end

assert(cfg.data_split_parts<20,'Over 20 data splits, check your data!'); % sanity check

% final upper limit estimate
cfg.max_mem_MB = max(ceil(mem_estimate_MB*1.25),5000); % min 5GB, 25% overhead

fprintf('\nEstimated memory trace per worker %0.1fGB (%i splits), requesting %0.1fGB (%s)\n',mem_estimate_MB/1000,cfg.data_split_parts,cfg.max_mem_MB/1000,datestr(datetime('now')));
    
%% Stage 1: Compute real unpermuted values (one job only)
cfg.stage1_cfgfiles = [cfg.workfolder,filesep,sprintf('stage1_cfg.mat')];
cfg.stage1_jobfiles = [cfg.workfolder,filesep,sprintf('stage1_jobfile')];
cfg.stage1_resultfiles = [cfg.workfolder,filesep,sprintf('stage1_results.mat')];

stage = 1; % current stage stored into cfg's
fprintf('\nStage 1: Unpermuted results computing (%s)\n',datestr(datetime('now')));
if ~exist(cfg.stage1_resultfiles,'file') || cfg.overwrite>0
    save(cfg.stage1_cfgfiles,'cfg','stage');        
    [jobname,logname] = sendjob(cfg.max_mem_MB,cfg.stage1_jobfiles,cfg.stage1_cfgfiles,cfg.worker_path,'bramila_ISC_worker',cfg.doLocalSerial);
    fprintf('...job submitted, waiting (%s)...\n',datestr(datetime('now')));
    wait_for_jobs({cfg.stage1_cfgfiles},{jobname},{cfg.stage1_jobfiles},{logname},2,true);
end

%% STAGE 2: compute permutations
stage = 2; % current stage stored into cfg's
for i = 1:cfg.parallel_jobs
    cfg.stage2_cfgfiles{i} = [cfg.workfolder,filesep,sprintf('stage2_permutation_cfg_set%i.mat',i)];
    cfg.stage2_jobfiles{i} = [cfg.workfolder,filesep,sprintf('stage2_permutation_jobfile_set%i',i)];
    cfg.stage2_resultfiles{i} = [cfg.workfolder,filesep,sprintf('stage2_results_set%i.mat',i)];        
    save(cfg.stage2_cfgfiles{i},'cfg','stage','-v7.3');
end
for i = 1:cfg.parallel_jobs
    cfg.process_index = i;
    save(cfg.stage2_cfgfiles{i},'cfg','stage','-v7.3');
end
cfg.process_index = [];

fprintf('\nStage 2: Permutation statistics computing (%s)\n',datestr(datetime('now')));

tic();
jobnames = cell(1,cfg.parallel_jobs);
lognames = jobnames;
wait_for=true(1,cfg.parallel_jobs);
for i = 1:cfg.parallel_jobs
    if ~exist(cfg.stage2_resultfiles{i},'file') || cfg.overwrite>0
        [jobnames{i},lognames{i}] = sendjob(cfg.max_mem_MB,cfg.stage2_jobfiles{i},cfg.stage2_cfgfiles{i},cfg.worker_path,'bramila_ISC_worker',cfg.doLocalSerial);
    else
        wait_for(i)=false;
        stage=3;
        save(cfg.stage2_cfgfiles{i},'stage','-append');
    end
end
fprintf('... %i jobs submitted (%s), waiting...\n',sum(wait_for),datestr(datetime('now')));
% note, we allow some jobs to fail here as not all permutation sets are needed
wait_for_jobs(cfg.stage2_cfgfiles(wait_for),jobnames(wait_for),cfg.stage2_jobfiles(wait_for),lognames(wait_for),3,false);

t=toc();
fprintf('Permutations completed in %0.1fmin with %0.2fs/permutation (%s)\n',t/60,t/(cfg.iter*cfg.parallel_jobs),datestr(now,'HH:MM:SS'));

% pool all nullvals that are available
nulldata = struct();
nulldata.exceedances_tfce_maxstat = 0;
nulldata.exceedances_tfce = 0;
nulldata.exceedances_raw = 0;
nulldata.exceedances_raw_maxstat = 0;
nulldata.cluster_max_sizes = [];
nulldata.perm_rates = [];
missing=0;
for i = 1:cfg.parallel_jobs
    if exist(cfg.stage2_resultfiles{i},'file')
        result = load(cfg.stage2_resultfiles{i});
        if result.failed_status==0   % data is valid     
            nulldata.exceedances_tfce_maxstat = nulldata.exceedances_tfce_maxstat + result.exceedances_tfce_maxstat;
            nulldata.exceedances_tfce = nulldata.exceedances_tfce + result.exceedances_tfce;
            nulldata.exceedances_raw = nulldata.exceedances_raw + result.exceedances_raw;
            nulldata.exceedances_raw_maxstat = nulldata.exceedances_raw_maxstat + result.exceedances_raw_maxstat;
            nulldata.cluster_max_sizes = [nulldata.cluster_max_sizes;result.cluster_max_sizes];
            nulldata.perm_rates = [nulldata.perm_rates,result.perm_rate];
        else
            missing=missing+1;
        end
    else
        missing=missing+1;
    end
end
Niter = length(nulldata.cluster_max_sizes);

assert(nnz(isnan(nulldata.cluster_max_sizes))==0,'NaN values found in pooled nullvalues, something went wrong!');

fprintf('Obtained total %i nullvalues after pooling (%i missing sets), worker average rate was %.2fs/permutation\n',Niter,missing,mean(nulldata.perm_rates));

nullfile = [cfg.workfolder,filesep,'pooled_nulldata.mat'];
save(nullfile,'nulldata','missing','-v7.3');          

% Now we have permutation maps and real maps, generate outputs
fprintf('\nPopulating output struct (%s)\n',datestr(now,'HH:MM:SS'));

realdata = load(cfg.stage1_resultfiles);

results=[];
results.raw_tval_map = realdata.REAL_tvals_map;
results.raw_tfce_map = realdata.REAL_tfce_map; % this is signed
results.mask = mask;
results.mask_ind = mask_ind;
results.cfg_struct = cfg;

% handle tfce maps
corrected = nulldata.exceedances_tfce_maxstat/Niter;
corrected_tfce_pval_map = ones(sz_mask);
corrected_tfce_pval_map(mask_ind) = corrected;
results.stats.tfce_pval_corrected = corrected_tfce_pval_map;

uncorrected = nulldata.exceedances_tfce/Niter;
uncorrected_tfce_pval_map = ones(sz_mask);
uncorrected_tfce_pval_map(mask_ind) = uncorrected;
results.stats.tfce_pval_uncorrected = uncorrected_tfce_pval_map;

% sanity check!
assert(nnz(uncorrected_tfce_pval_map>corrected_tfce_pval_map)==0);

% handle raw value maps
uncorrected = nulldata.exceedances_raw/Niter;
uncorrected_raw_pval_map = ones(sz_mask);
uncorrected_raw_pval_map(mask_ind) = uncorrected;
results.stats.raw_pval_uncorrected = uncorrected_raw_pval_map;

% FWE maxstat corrected data
corrected = nulldata.exceedances_raw_maxstat/Niter;
corrected_raw_pval_map = ones(sz_mask);
corrected_raw_pval_map(mask_ind) = corrected;
results.stats.raw_pval_FWE = corrected_raw_pval_map;

% FDR corrected data
corrected=mafdr(uncorrected,'BHFDR','True');
corrected_raw_pval_map = ones(sz_mask);
corrected_raw_pval_map(mask_ind) = corrected;
results.stats.raw_pval_FDR = corrected_raw_pval_map;

% sanity check!
assert(nnz(uncorrected_raw_pval_map>corrected_raw_pval_map)==0);

% handle cluster extend maps
clust_sizes = unique(realdata.REAL_clstat(realdata.REAL_clstat>0));
corrected_cluster_pval=ones(length(realdata.REAL_clstat),1);
for csize=clust_sizes(:)'
    p = sum(nulldata.cluster_max_sizes>=csize)/Niter;
    corrected_cluster_pval(realdata.REAL_clstat==csize)=p;
end
corrected_cluster_pval_map = ones(sz_mask);
corrected_cluster_pval_map(mask_ind) = corrected_cluster_pval;
results.stats.cluster_pval_corrected = corrected_cluster_pval_map;
results.stats.cluster_voxelwise_extent = realdata.REAL_clstat_map;

results.stats.cluster_thresholds.p005FWE = ceil(prctile(nulldata.cluster_max_sizes,95));
results.stats.cluster_thresholds.p001FWE = ceil(prctile(nulldata.cluster_max_sizes,99));
results.stats.cluster_thresholds.p0001FWE = ceil(prctile(nulldata.cluster_max_sizes,99.9));

fprintf('\nResult preview: Voxels at corrected p<0.01 (pos+neg):\n... %i+%i for voxel-wise FDR\n... %i+%i for voxel-wise FWE\n... %i+%i for TFCE FWE\n... %i+%i for cluster extend FWE\n',...
    nnz(results.stats.raw_pval_FDR<0.01 & results.raw_tval_map>0),nnz(results.stats.raw_pval_FDR<0.01 & results.raw_tval_map<0),...
    nnz(results.stats.raw_pval_FWE<0.01 & results.raw_tval_map>0),nnz(results.stats.raw_pval_FWE<0.01 & results.raw_tval_map<0),...
    nnz(results.stats.tfce_pval_corrected<0.01 & results.raw_tval_map>0),nnz(results.stats.tfce_pval_corrected<0.01 & results.raw_tval_map<0),...
    nnz(results.stats.cluster_pval_corrected<0.01 & results.raw_tval_map>0),nnz(results.stats.cluster_pval_corrected<0.01 & results.raw_tval_map<0));

resultfile = [cfg.workfolder,filesep,'final_results.mat'];
save(resultfile,'cfg','results','-v7.3');          


fprintf('\n---- All finished (%s) ----\n',datestr(now,'HH:MM:SS'));

diary off

end

function [jobname,logfile] = sendjob(max_mem,filename,paramfile,codepath,funfile,doLocal)
% write and send job (or run locally)

if nargin<6
    doLocal=0;
end

logfile = [filename,'_log'];

dlmwrite(filename, '#!/bin/sh', '');
dlmwrite(filename, '#SBATCH -p batch','-append','delimiter','');
dlmwrite(filename, '#SBATCH -t 04:00:00','-append','delimiter','');
dlmwrite(filename, '#SBATCH -N 1','-append','delimiter','');
dlmwrite(filename, '#SBATCH -n 1','-append','delimiter','');
dlmwrite(filename, '#SBATCH --qos=normal','-append','delimiter','');
dlmwrite(filename, ['#SBATCH -o "' logfile '"'],'-append','delimiter','');
dlmwrite(filename, sprintf('#SBATCH --mem-per-cpu=%i',max_mem),'-append','delimiter','');
dlmwrite(filename, 'hostname; date;','-append','delimiter','');
dlmwrite(filename, 'module load matlab','-append','delimiter','');
dlmwrite(filename,sprintf('srun matlab -nosplash -nodisplay -nodesktop -r "cd(''%s'');fprintf('' current path: %%s '',pwd());%s(''%s'');exit;"',codepath,funfile,paramfile),'-append','delimiter','');

jobname = 'unknown';

if doLocal==1
    %%%% FOR TESTING AND DEBUGGING ONLY - run locally in serial manner
    command = sprintf('%s(''%s'');',funfile,paramfile);
    eval(command);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    [a,b]=unix(['sbatch ' filename]);
    s = 'Submitted batch job ';
    k = strfind(b,s);
    if ~isempty(k)
        jobname = strtrim(b(length(s):end));
    end
end

end

function wait_for_jobs(CONFIGFILES,jobnames,jobfiles,lognames,STAGE,MUST_FINISH)
% This function monitors progress of jobs and resubmits failed jobs if
% necessary. Status of each cfg file must be valid to mark it completed. If
% there are too many failed submissions, we crash (check logs to solve).
if ~iscell(CONFIGFILES)
    temp{1} = CONFIGFILES;
    CONFIGFILES = temp;
end
N_files = length(CONFIGFILES);
if N_files==0
    return;
end
done_files = zeros(1,N_files);
resubmit_count = zeros(1,N_files);
failed_files = zeros(1,N_files);

% parameters for job monitoring
POLL_DELAY = 20;
TIMES_TO_TRY = 3;
RESUBMIT_DELAY = 30;
PRINT_DELAY = 5*60; %

MAX_HOURS_TO_RUN = 12;

resubmit_delays = zeros(1,N_files);
starttime = tic();
NEXT_PRINTOUT = 0;

while 1
    pause(POLL_DELAY);    
    % total time
    TOT_time = toc(starttime);    
    for i = 1:N_files
        done_files(i) = check_stage(i,CONFIGFILES,STAGE);
    end
    for i = 1:N_files        
        if done_files(i)==0 && failed_files(i)==0
            FAILED=0;
            [job_state,job_message]=unix(['qstat -f ',jobnames{i}]);            
            if job_state==153
                FAILED = 1; % job is not complete and not in queue, it has failed
            end            
            if FAILED>0
                % due to delays, check the status again
                done_files(i) = check_stage(i,CONFIGFILES,STAGE);
                if done_files(i)==0
                    if resubmit_count(i)<TIMES_TO_TRY
                        d = resubmit_delays(i)+toc(starttime);
                        if d > RESUBMIT_DELAY
                            resubmit_count(i)=resubmit_count(i)+1;
                            fprintf('FAILED with state code %i, resubmitting job ''%s'' (message: %s, %ith time, delay %is)\n',job_state,jobfiles{i},job_message,resubmit_count(i),round(d));
                            if exist(lognames{i},'file') % copy old file if available (for debugging)
                                copyfile(lognames{i},[lognames{i},'_failed_nr',num2str(resubmit_count(i))]);
                            end
                            [~,jobnames{i}] = system(['sbatch ' jobfiles{i}]);
                            resubmit_delays(i)=-toc(starttime);
                        end
                    else
                        failed_files(i)=1;
                        warning('Tried to submit job %s over %i times, job failed!!',jobfiles{i},TIMES_TO_TRY);
                    end
                end
            end
        end
    end
    
    if sum(done_files)==length(done_files)
        break;
    end
    % make sure the loop stops after specific time
    if TOT_time/60/60 > MAX_HOURS_TO_RUN
        warning('Aborting after waiting for jobs to finish (TOT_time=%fs)',TOT_time);
        break;
    end
    if TOT_time > NEXT_PRINTOUT
        fprintf('... [%s] %i jobs completed, %i resubmitted (%i jobs in total)\n',datestr(datetime('now')),nnz(done_files),nnz(resubmit_count),N_files);
        NEXT_PRINTOUT = toc(starttime) + PRINT_DELAY;
    end
end

if MUST_FINISH % do we need all results in order to continue processing?
    assert(sum(done_files)==length(done_files),'Some jobs failed to complete! Not allowed!')
end

fprintf('... All %i jobs finished (%i failed)!\n',nnz(done_files),sum(failed_files));

end

function ISDONE = check_stage(i,CONFIGFILES,STAGE)
ISDONE=0;
if exist(CONFIGFILES{i},'file')>0
    for try_loop=1:3 % try few times in case there is simultaneous writing
        try
            load(CONFIGFILES{i});
            if stage>=STAGE
                ISDONE = 1;
                return;
            else
                return;
            end
        catch err
            pause(1);
        end
    end
end

end
