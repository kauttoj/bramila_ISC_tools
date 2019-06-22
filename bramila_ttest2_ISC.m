function results=bramila_ttest2_ISC(cfg)
% results = bramila_mantel_ttest2_ISC(cfg)
%
% Whole brain element-wise two-sample t-test for ISC similarity matrices. Input 4D nifti file
% needs to contain the upper triangle elements of the similarity matrix
% (same format as the output of ISC toolbox with option "store
% correlation matrices"). Statistics include standard FDR, TFCE and cluster extent.
%
% Input parameters:
%
%   cgf.infile= 4D volume or path to NIFTI, every volume (4th index) is a pair of subjects' similarity (MANDATORY)
%   cfg.mask = 3D brain mask or path to NIFTI, all values >0 are included (MANDATORY)
%   cfg.group_id = vector of group membership, only values 1,2 and 0 are allowed. Must match total number of subjects in ISC matrix! (MANDATORY)
%   cfg.p_val_threshold = cluster forming threshold p-value in using parametric test (OPTIONAL)
%   cfg.NumWorkers = number of requested workers in parfor (default from the local profile) (OPTIONAL)
%   cfg.iter = number of permutations (default 5000) (OPTIONAL)
%   cfg.permutation_type = how to permute data, 'subjectwise' = permute rows/cols (conservative, default), 'elementwise' = permute elements (liberal, unrestricted permutations)
%   cfg.doFisherTransform = do Fisher transform for ISC values (default 1) (OPTIONAL), skip if already converted!
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
%          results.stats.raw_pval_corrected: FDR corrected p-vals
%      results.stats.cluster_pval_corrected: cluster extent corrected p-vals (repeated over cluster)
%    results.stats.cluster_voxelwise_extent: number of voxels in the corresponding cluster
%          results.stats.cluster_thresholds: list of voxels extends for p<{0.05,0.01,0.001} used in correction                      
%
% NOTE: All results are two-tailed
%
%  J. Kauttonen, M. Hakonen, E. Glerean
%  2018, Brain and Mind Laboratory Aalto University
%
%  29.11.2018 first version
%  22.6.2019 added subjectwise permutations (stricker test)

% if no input is given, create and analyze dummy test data
if nargin==0,    
    cfg=[];    
    cfg.group_id = [0,ones(1,10),0,0,2*ones(1,10),0];
    N_subj = length(cfg.group_id);       
    cfg.mask = zeros(41,45,47);
    cfg.mask(5:38,12:42,2:41)=1;
    cfg.mask(1,1,1)=1;
    N_pairs = N_subj*(N_subj-1)/2;
    cfg.infile= randn(41,45,47,N_pairs);
    inds = find(triu(ones(N_subj,N_subj),1));
    model1 = corr(randn(500,N_subj));
    model1(1:12,1:12)=1;
    model2 = corr(randn(500,N_subj));
    model2(13:end,13:end)=1;    
    % create one big high-correlation cube
    total_significant_pos = 0;
    total_significant_neg = 0;
    for x=22:30
        for y=29:39
            for z=10:30                
                nullmodel = corr(randn(500,N_subj));
                if z<20 % use model1
                    cfg.infile(x,y,z,:)=0.30*model1(inds) + 0.70*nullmodel(inds);
                    total_significant_pos=total_significant_pos+1;
                else % use model2
                    cfg.infile(x,y,z,:)=0.30*model2(inds) + 0.70*nullmodel(inds);
                    total_significant_neg=total_significant_neg+1;
                end
            end
        end
    end
    % single "wild voxel" with very high correlation, but no neighbors
    cfg.infile(1,1,1,:) = 0.50*model2(inds) + 0.50*nullmodel(inds);    
    %cfg.modelNI = [];
    cfg.iter = 700;    
    cfg.NumWorkers=1;
    cfg.doFisherTransform=0;
    RESULTS=bramila_ttest2_ISC(cfg);  
    return
end

fprintf('---- Running "bramila_ttest2_ISC" two-sample t-test for volumetric data (%s) ----\n',datestr(now,'HH:MM:SS'));

% fix seed for repeatability
rng(666);

% p-val threshold for cluster extend correction
if ~isfield(cfg,'p_val_threshold')
   cfg.p_val_threshold = 0.01;
end
assert(cfg.p_val_threshold<=0.01); % do not allow weaker thresholds

if ~isfield(cfg,'doFisherTransform')
   cfg.doFisherTransform = 1;
end

% start pool or use existing
mycluster = gcp('nocreate');
if isempty(mycluster)
    mycluster=parcluster;
    if isfield(cfg,'NumWorkers')
        mycluster.NumWorkers = max(1,cfg.NumWorkers);
    end
    parpool(mycluster);
end
cfg.NumWorkers = mycluster.NumWorkers;

% permutation type
if ~isfield(cfg,'permutation_type')
    cfg.permutation_type='subjectwise';
end
assert(sum(ismember(cfg.permutation_type,{'subjectwise','elementwise'}))>0,'Unknown permutation type!');

% initialize iter
if(~isfield(cfg,'iter'))
    cfg.iter=5000;
end
cfg.iter = max(cfg.iter,10);
assert(cfg.iter<50000); % too much will likely lead to memory issues

if cfg.iter<5000
   warning('!! Only %i iterations used, results are unreliable !!',cfg.iter);
end

% group indices of rows & columns
group1_ind = find(cfg.group_id==1);
group2_ind = find(cfg.group_id==2);
group_rows = find(cfg.group_id==1 | cfg.group_id==2);
Nsubj1 = length(group1_ind);
Nsubj2 = length(group2_ind);
Nsubs = Nsubj1+Nsubj2;

if Nsubj1<4 || Nsubj2<4
    error('Cannot continue with only %i+%i subjects!',Nsubj1,Nsubj2);
end

% load data
if ischar(cfg.infile)
    sprintf('Loading ISC file %s\n',cfg.infile);
    nii=load_nii(cfg.infile);
    data = nii.img;
    clear nii;
else
    % data is already matrix
    data = cfg.infile;
    cfg.infile=[];
end

if cfg.doFisherTransform
    numNans = nnz(isnan(data));
    assert(nanmax(data(:))<1 && nanmin(data(:))>-1,'Data must be -1<x<1 before Fisher transform!');
    data = atanh(data);
    assert(numNans == nnz(isnan(data)),'Fisher transformation failed, NaN values emerged! Check your input data!');
end
assert(isreal(data),'Data must be real-valued!');

% load mask
if ischar(cfg.mask)
    sprintf('Loading maskfile %s\n',cfg.mask);
    nii=load_nii(cfg.mask);
    mask=nii.img;
    clear nii;
else
    mask = cfg.mask;
    cfg.mask=[];
end
mask = mask>0;
sz_mask = size(mask);

% make sure mask makes sense for a typical experiment
mask_coverage_prc = 100*nnz(mask)/numel(mask);
assert(mask_coverage_prc>5 && mask_coverage_prc<50);
% check dimensions
for i=1:3
    assert(sz_mask(i)==size(data,i));
end

inmask=find(mask>0);
Nvoxels = length(inmask);
NISCvalues = size(data,4);
Nsubs_ISC = (2*NISCvalues + 1/4)^(1/2) + 1/2;

assert(Nsubs_ISC==length(cfg.group_id)); % these need to match
assert(round(Nsubs_ISC)==Nsubs_ISC);

% check that ISC value and subject counts make sense
assert(Nsubs_ISC >= Nsubs);

% ISC matrix indices - THIS IS THE STANDARD FOR ISC TOOLBOX!
ISC_mat_inds=find(triu(ones(Nsubs_ISC),1));

% make full index matrix
index_mat = zeros(Nsubs_ISC);
index_mat(ISC_mat_inds)=1:length(ISC_mat_inds);
index_mat = index_mat + index_mat';

% make group-wise index matrices
group_mat = zeros(Nsubs_ISC);
group_mat(group1_ind,group1_ind)=1;
group_mat(group2_ind,group2_ind)=2;
ind1=index_mat(triu(group_mat==1,1));
ind2=index_mat(triu(group_mat==2,1));
ISC_mat_inds_allgroups = [ind1',ind2']; % this extracts all elements of groups

% which indices belong to which groups
ISC_mat_inds_group1 = 1:length(ind1);
ISC_mat_inds_group2 = ISC_mat_inds_group1(end) + (1:length(ind2));

% options for TFCE, these are the defaults for 3D fMRI data
tfce_opts=[];
tfce_opts.tfce.H              = 2;                  % TFCE H parameter
tfce_opts.tfce.E              = 0.5;                % TFCE E parameter
tfce_opts.tfce.conn           = 26;                 % TFCE connectivity neighbourhood (26 all neighbors in 3D)
tfce_opts.tfce.deltah         = 0;             % Delta-h of the TFCE equation (0 = auto with 100 steps)
cfg.tfce_opts = tfce_opts;

% reorder data
data=reshape(data,[],length(ISC_mat_inds));
data = data(inmask,:);
data=data';

% compute real, unpermuted values
[~,pvals,~,stats] = ttest2(...
    data(ISC_mat_inds_allgroups(ISC_mat_inds_group1),:),...
    data(ISC_mat_inds_allgroups(ISC_mat_inds_group2),:),...
    0.05,'both','unequal');
REAL_tvals = stats.tstat;
assert(nnz(isnan(REAL_tvals))==0 && nnz(~isfinite(REAL_tvals))==0);

REAL_tvals_map = zeros(sz_mask);
REAL_tvals_map(inmask) = REAL_tvals;
pvals_map = ones(sz_mask);
pvals_map(inmask) = pvals; % lower = better
[~,REAL_clstat,REAL_sizes] = palm_clustere_vol(1-pvals_map,1-cfg.p_val_threshold,inmask);

REAL_clstat_map = zeros(sz_mask);
REAL_clstat_map(inmask) = REAL_clstat;

REAL_tfce_vals_signed = palm_tfce(REAL_tvals_map.*(REAL_tvals_map>0),tfce_opts,inmask) - palm_tfce(-REAL_tvals_map.*(REAL_tvals_map<0),tfce_opts,inmask);
REAL_tfce_vals = abs(REAL_tfce_vals_signed);
REAL_tfce_map = zeros(sz_mask);
REAL_tfce_map(inmask) = REAL_tfce_vals_signed;

fprintf('Input data: Volume size %i*%i*%i with %i masked voxels (%0.1f%% coverage) and %i+%i subjects (total %i in ISC matrix)\n',...
    sz_mask(1),sz_mask(2),sz_mask(3),Nvoxels,mask_coverage_prc,Nsubj1,Nsubj2,Nsubs_ISC);

% create null
exceedances_tfce_maxstat = zeros(1,Nvoxels);
exceedances_tfce = zeros(1,Nvoxels);
exceedances_raw = zeros(1,Nvoxels);
cluster_max_sizes = nan(cfg.iter,1);

% create permutated groups, first permutation is the real order!
assert(Nsubs_ISC<250,'Over 250 subjects, cannot use uint16 array!'); % uint16 limit check
fprintf('Creating permutation sets (%s)\n',datestr(now,'HH:MM:SS'));    

iter=1;
L=length(ISC_mat_inds_allgroups);
unordered = 1:length(ISC_mat_inds_allgroups);
unordered_sub = 1:Nsubs;
perms_sets = zeros(L,cfg.iter,'uint16');
th = max(1,round(0.05*L));
th_sub = max(1,round(0.05*Nsubs));

testdata =  meshgrid(1:Nsubs_ISC,1:Nsubs_ISC);

while iter < cfg.iter+1
    % require that least 5% of elements are permuted (could be even stricker!)
    if strcmp(cfg.permutation_type,'subjectwise')
        % permute subject identities (restricted permutation sets)
        perm = randperm(Nsubs);                       
        if nnz(perm-unordered_sub)<th_sub
            continue;
        end       
        arr = 1:Nsubs_ISC;
        arr(group_rows) = group_rows(perm);                 
        index_mat_perm = index_mat(arr,arr);                        
        ind1=index_mat_perm(triu(group_mat==1,1));
        ind2=index_mat_perm(triu(group_mat==2,1));
        perm_set = [ind1',ind2'];        
    else % elementwise
        % permute elements (unrestricted permutation sets)
        perm = randperm(L);
        if nnz(perm-unordered)<th
            continue;
        end        
        perm_set = ISC_mat_inds_allgroups(perm);
    end
    perms_sets(:,iter)=perm_set;
    iter=iter+1;
end
assert(nnz(perms_sets==0)==0,'zeros in permutation indices, BUG!');

fprintf('Starting permutations with %i iterations and %i workers (%s)\n',cfg.iter,cfg.NumWorkers,datestr(now,'HH:MM:SS'))
pause(0.05); % pause to allow printing
tic;

parfor (iter = 1:cfg.iter,cfg.NumWorkers)
%for iter = 1:cfg.iter           
    % get permuted model
    null_inds = perms_sets(:,iter);    
    [~,pvals,~,stats] = ttest2(...
        data(null_inds(ISC_mat_inds_group1),:),...
        data(null_inds(ISC_mat_inds_group2),:),...
        0.05,'both','unequal');
    tvals = stats.tstat;
    assert(nnz(isnan(tvals))==0 && nnz(~isfinite(tvals))==0);
    
    tvals_map = zeros(sz_mask);
    tvals_map(inmask) = tvals;
    
    pvals_map = zeros(sz_mask);
    pvals_map(inmask) = 1-pvals; % higher = better
    
    % get maximum cluster of both tails (makes no sense to merge positive and negative)
    cluster_max_sizes(iter) = max(...
        palm_clustere_vol(pvals_map.*(tvals_map>0),1-cfg.p_val_threshold,inmask),...
        palm_clustere_vol(pvals_map.*(tvals_map<0),1-cfg.p_val_threshold,inmask)); 
        
    % get voxel-wise (uncorrected stats)
    curexceeds = abs(tvals) >= abs(REAL_tvals);
    exceedances_raw = exceedances_raw + curexceeds;
    
    % make tfce map (all positive!)
    tfce_vals = palm_tfce(tvals_map.*(tvals_map>0),tfce_opts,inmask) + palm_tfce(-tvals_map.*(tvals_map<0),tfce_opts,inmask);
    % get max stats
    curexceeds = max(tfce_vals) >= REAL_tfce_vals;
    exceedances_tfce_maxstat = exceedances_tfce_maxstat + curexceeds;
    % get voxel-wise (uncorrected stats)
    curexceeds = tfce_vals >= REAL_tfce_vals;
    exceedances_tfce = exceedances_tfce + curexceeds;
    
end
assert(nnz(isnan(cluster_max_sizes))==0);

t = toc();
fprintf('Permutations completed in %0.1fmin with %0.2fs/permutation (%s)\n',t/60,t/cfg.iter,datestr(now,'HH:MM:SS'));
fprintf('Populating output struct\n');

results=[];
results.raw_tval_map = REAL_tvals_map;
results.raw_tfce_map = REAL_tfce_map; % this is signed
results.mask = mask;
results.mask_ind = inmask;
results.cfg_struct = cfg;

% handle tfce maps
corrected = exceedances_tfce_maxstat/cfg.iter;
corrected_tfce_pval_map = ones(sz_mask);
corrected_tfce_pval_map(inmask) = corrected;
results.stats.tfce_pval_corrected = corrected_tfce_pval_map;

uncorrected = exceedances_tfce/cfg.iter;
uncorrected_tfce_pval_map = ones(sz_mask);
uncorrected_tfce_pval_map(inmask) = uncorrected;
results.stats.tfce_pval_uncorrected = uncorrected_tfce_pval_map;

% sanity check!
assert(nnz(uncorrected_tfce_pval_map>corrected_tfce_pval_map)==0);

% handle raw value maps
uncorrected = exceedances_raw/cfg.iter;
uncorrected_raw_pval_map = ones(sz_mask);
uncorrected_raw_pval_map(inmask) = uncorrected;
results.stats.raw_pval_uncorrected = uncorrected_raw_pval_map;

corrected=mafdr(uncorrected,'BHFDR','True');
corrected_raw_pval_map = ones(sz_mask);
corrected_raw_pval_map(inmask) = corrected;
results.stats.raw_pval_corrected = corrected_raw_pval_map;

% sanity check!
assert(nnz(uncorrected_raw_pval_map>corrected_raw_pval_map)==0);

% handle cluster extend maps
clust_sizes = unique(REAL_clstat(REAL_clstat>0));
corrected_cluster_pval=ones(length(REAL_clstat),1);
for csize=clust_sizes(:)'
    p = sum(cluster_max_sizes>=csize)/cfg.iter;
    corrected_cluster_pval(REAL_clstat==csize)=p;
end
corrected_cluster_pval_map = ones(sz_mask);
corrected_cluster_pval_map(inmask) = corrected_cluster_pval;
results.stats.cluster_pval_corrected = corrected_cluster_pval_map;
results.stats.cluster_voxelwise_extent = REAL_clstat_map;

results.stats.cluster_thresholds.p005FWE = ceil(prctile(cluster_max_sizes,95));
results.stats.cluster_thresholds.p001FWE = ceil(prctile(cluster_max_sizes,99));
results.stats.cluster_thresholds.p0001FWE = ceil(prctile(cluster_max_sizes,99.9));

fprintf('Voxels at corrected p<0.01 (pos+neg): %i+%i (voxel-wise FDR), %i+%i (TFCE), %i+%i (cluster extend)\n',...
    nnz(results.stats.raw_pval_corrected<0.01 & results.raw_tval_map>0),nnz(results.stats.raw_pval_corrected<0.01 & results.raw_tval_map<0),...
    nnz(results.stats.tfce_pval_corrected<0.01 & results.raw_tval_map>0),nnz(results.stats.tfce_pval_corrected<0.01 & results.raw_tval_map<0),...
    nnz(results.stats.cluster_pval_corrected<0.01 & results.raw_tval_map>0),nnz(results.stats.cluster_pval_corrected<0.01 & results.raw_tval_map<0));

fprintf('---- All finished (%s) ----\n',datestr(now,'HH:MM:SS'));

end

% this is a stripped version from the palm function
function tfcestat = palm_tfce(D,opts,mask)
% Compute the TFCE statistic, for volume or surface
% data (vertexwise or facewise).
%
% Usage:
% tfcestat = palm_tfce(X,y,opts,plm)
%
% Inputs:
% - X    : Statistical map.
% - y    : Modality index (of those stored in the plm struct).
% - opts : Struct with PALM options.
% - plm  : Struct with PALM data.
%
% Outputs:
% - tfcestat  : TFCE map.
%
% _____________________________________
% Anderson M. Winkler
% FMRIB / University of Oxford
% Sep/2013
% http://brainder.org

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% PALM -- Permutation Analysis of Linear Models
% Copyright (C) 2015 Anderson M. Winkler
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Inject the data.
X = D(mask);

% Split the code according to whether dh is "automatic" or a fixed
% value given supplied by the user.
if opts.tfce.deltah == 0,
    % "delta h"
    dh = max(X(:))/100;    
    % Volume (voxelwise data)
    tfcestat = zeros(size(D));
    for h = dh:dh:max(D(:))
        CC    = bwconncomp(D>=h,opts.tfce.conn);
        integ = cellfun(@numel,CC.PixelIdxList).^opts.tfce.E * h^opts.tfce.H;
        for c = 1:CC.NumObjects,
            tfcestat(CC.PixelIdxList{c}) = ...
                tfcestat(CC.PixelIdxList{c}) + integ(c);
        end
    end        
else
    % "delta h"
    dh = opts.tfce.deltah;
    
    % Volume (voxelwise data)
    tfcestat  = zeros(size(D));
    h         = dh;
    CC        = bwconncomp(D>=h,opts.tfce.conn);
    while CC.NumObjects,
        integ = cellfun(@numel,CC.PixelIdxList).^opts.tfce.E * h^opts.tfce.H;
        for c = 1:CC.NumObjects,
            tfcestat(CC.PixelIdxList{c}) = ...
                tfcestat(CC.PixelIdxList{c}) + integ(c);
        end
        h     = h + opts.tfce.deltah;
        CC    = bwconncomp(D>=h,opts.tfce.conn);
    end
    
end

% Return as a vector with the same size as X, and
% apply the correction for the dh.
tfcestat = tfcestat(mask);
tfcestat = tfcestat(:)' * dh;

end

% this is a stripped version from the palm function
function [maxsize,clstat,sizes] = palm_clustere_vol(D,thr,mask)
% Compute cluster extent statistics, for volume or surface
% data (vertexwise or facewise).
% 
% Usage:
% [maxsize,clstat,sizes] = palm_clustere(X,y,thr,opts,plm)
% 
% Inputs:
% - X    : Statistical map.
% - y    : Modality index (of those stored in the plm struct).
% - thr  : Cluster-forming threshold.
% - opts : Struct with PALM options.
% - plm  : Struct with PALM data.
% 
% Outputs:
% - maxsize : Largest cluster extent.
% - clstat  : Thresholded map with the cluster sizes (cluster statistic).
% - sizes   : Vector with all cluster sizes.
% 
% _____________________________________
% Anderson M. Winkler
% FMRIB / University of Oxford
% Sep/2013
% http://brainder.org

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% PALM -- Permutation Analysis of Linear Models
% Copyright (C) 2015 Anderson M. Winkler
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Dt      = D > thr;
% Compute the sizes and the statistic
% Connected components: bwconncomp is slightly faster and
% less memory intensive than bwlabel
CC = bwconncomp(Dt);

% Here cellfun is ~30% faster than doing as a loop
sizes = cellfun(@numel,CC.PixelIdxList);

% Compute the statistic image (this should be for the 1st perm only)
if nargout > 1,
    clstat = zeros(size(Dt));
    for c = 1:CC.NumObjects,
        clstat(CC.PixelIdxList{c}) = sizes(c);
    end
    clstat = clstat(mask)';
end
 
% In fact, only the max matters, because the uncorrected cluster extent
% doesn't make much sense. Make sure the output isn't empty.
if isempty(sizes),
    sizes = 0;
end
maxsize = max(sizes);

end

% % t-value with unequal variance assumption
% function tval=tt_np(data,g1,g2)
% 
%     nx = length(g1);
%     ny = length(g2);
% 
%     difference = mean(data(:,g1),2) - mean(data(:,g2),2);
% 
%     s2x = var(data(:,g1),[],2);
%     s2y = var(data(:,g2),[],2);
%     s2xbar = s2x ./ nx;
%     s2ybar = s2y ./ ny;
%     se = sqrt(s2xbar + s2ybar);
%     tval = difference ./ se;
%     
% end