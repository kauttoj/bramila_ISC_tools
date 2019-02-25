function results=bramila_supervised_ISC(cfg)
% results=bramila_supervised_ISC(cfg)
%
% Whole supervised (regression and classification) analysis for ISC similarity matrices using k-nearest neighbors learner.
% Input 4D nifti file needs to contain the upper triangle elements of the similarity matrix
% (same format as the output of ISC toolbox with option "store correlation matrices"). 
% Statistics include standard FDR, TFCE and cluster extent.
%
% Input parameters:
%
%   cgf.infile= 4D volume or path to NIFTI, every volume (4th index) is a pair of subjects' similarity (MANDATORY)
%     NOTE: These are similarity values, i.e., high ISC = high similarity. This cannot be automatically checked!
%   cfg.mask = 3D brain mask or path to NIFTI, all values >0 are included (MANDATORY)
%   cfg.target = target value of subject, one value per subject (rows of ISC matrices) (MANDATORY)
%   cfg.p_val_threshold = cluster forming threshold p-value in using parametric test (OPTIONAL)
%   cfg.NumWorkers = number of requested workers in parfor (default from the local profile) (OPTIONAL)
%   cfg.type = target type, must be regression or classification
%   cfg.iter = number of permutations (default 5000) (OPTIONAL)
%   cfg.k_vals = array of nearest neighbors, results are averaged over these (OPTIONAL)
%   cfg.distance_weighting = use distance weighting (0 or 1) (OPTIONAL)
%
% Output results struct with following fields:
%
%               results.raw_correlation_map: 3D voxel-wise correlation (regression) or accuracy (classification) values
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
%  J. Kauttonen
%  2018, Brain and Mind Laboratory Aalto University
%
%  18.2.2019 first version

% if no input is given, create and analyze test data
if nargin==0,    
    N_subj = 20;    
    cfg.mask = zeros(31,32,33);
    cfg.mask(5:25,12:32,5:20)=1;
    N_pairs = N_subj*(N_subj-1)/2;
    cfg.infile= randn(31,32,33,N_pairs);
    inds = find(triu(ones(N_subj,N_subj),1));
    
    cfg.type = 'regression';
    %cfg.type = 'classification';
    if strcmp(cfg.type,'regression')  
        group = randi(5,N_subj,1);
        factor = randn(500,5);
        cfg.target = group+0.1*rand(size(group));
    elseif strcmp(cfg.type,'classification')  
        group = randi(2,N_subj,1);
        factor = randn(500,2);        
        cfg.target = group;
    end        
    ind_triu = find(triu(ones(N_subj,N_subj),1));
    
    data = zeros(500,N_subj);
    for i=1:N_subj
        data(:,i)=factor(:,group(i));
    end           
    for x=1:size(cfg.mask,1),
        for y=1:size(cfg.mask,2),
            for z=1:size(cfg.mask,3),      
                if cfg.mask(x,y,z)==1
                    mat = corr(data+0.1*randn(500,N_subj));
                    cfg.infile(x,y,z,:)=mat(ind_triu);
                end
            end
        end
    end    
    
    %cfg.modelNI = [];
    cfg.iter = 100;
    cfg.NumWorkers=4;
    
    RESULTS=bramila_supervised_ISC(cfg);  
    return
end

fprintf('---- Running "bramila_supervised_ISC" test for volumetric data (%s) ----\n',datestr(now,'HH:MM:SS'));

% fix seed for repeatability
rng(666);

% p-val threshold for cluster extend correction
if ~isfield(cfg,'p_val_threshold')
   cfg.p_val_threshold = 0.01;
end
assert(cfg.p_val_threshold<=0.01); % do not allow threshold above 0.01

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

% initialize type
assert(ismember(cfg.type,{'regression','classification'}),'Problem type must be regression or classification');

% initialize iter
if(~isfield(cfg,'iter'))
    cfg.iter=5000;
end
cfg.iter=max(10,cfg.iter);
assert(cfg.iter<50000); % too much will likely lead to memory issues
if cfg.iter<5000
   warning('!! Only %i iterations used, results are unreliable !!',cfg.iter);
end

% distance weighting for samples
if(~isfield(cfg,'distance_weighting'))
    cfg.distance_weighting=1;
end

% k-values array
if(~isfield(cfg,'k_vals'))
    cfg.k_val=5; % 5 is the default in scikit-learn
end

%% starting
Nsubs = length(cfg.target);
cfg.target = cfg.target(:);

if strcmp(cfg.type,'classification')
    assert(length(unique(cfg.target))==2,'Only binary classification is supported at moment!');
end

if Nsubs<6
    error('Cannot continue with only %i subjects!',Nsubs)
end

% ISC matrix indices - THIS IS THE STANDARD FOR ISC TOOLBOX!
ISC_mat_inds=find(triu(ones(Nsubs),1));
NISCvalues = length(ISC_mat_inds);

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

assert(length(cfg.target)==Nsubs,'Target length must be equal to subjects!');

inmask=find(mask>0);
Nvoxels = length(inmask);

% check that ISC value and subject counts match
assert((2*NISCvalues + 1/4)^(1/2) + 1/2 == Nsubs);
assert(size(data,4) == NISCvalues);

% make index matrix
index_mat = zeros(Nsubs);
index_mat(ISC_mat_inds)=1:length(ISC_mat_inds);
index_mat = index_mat + index_mat';

% create permutated groups, first permutation is the real order!
assert(Nsubs<256); % uint8 limit
fprintf('Creating permutation sets (%s)\n',datestr(now,'HH:MM:SS'));    
if Nsubs<8
    perms_sets = perms(1:Nsubs);
    perms_sets(end,:)=[];
else
    iter=1;
    th = max(1,ceil(Nsubs*0.05));
    unordered = (1:Nsubs)';
    perms_sets = zeros(Nsubs,cfg.iter);
    while iter < cfg.iter+1
        % require that least 5% subjects are permuted (could be even stricker!)
        perm = randperm(Nsubs)';
        if nnz(perm-unordered)<th
            continue;
        end
        % do not repeat the same sets
        if iter>1 && any(sum(perm - perms_sets(:,1:(iter-1))==0)==Nsubs)
            continue;
        end        
        perms_sets(:,iter)=perm;
        iter=iter+1;
    end
end
assert(nnz(perms_sets==0)==0);
perms_sets = uint8(perms_sets); % reduce memory usage in loop

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

model_params.k_val = cfg.k_val;
model_params.isRegression = strcmp(cfg.type,'regression');
model_params.isWeighted = cfg.distance_weighting;

% compute real, unpermuted values
[REAL_corvals,pvals] = compute_accuracy(data,cfg.target,model_params);

assert(nnz(REAL_corvals<0 & pvals<0.05)==0,'negative tail contains p-values with p<0.05!')

REAL_corvals_map = zeros(sz_mask);
REAL_corvals_map(inmask) = REAL_corvals;
pvals_map = zeros(sz_mask); % these pvals are only for forming clusters
pvals_map(inmask) = 1-pvals;
[~,REAL_clstat,REAL_sizes] = palm_clustere_vol(pvals_map,1-cfg.p_val_threshold,inmask);

REAL_clstat_map = zeros(sz_mask);
REAL_clstat_map(inmask) = REAL_clstat;

REAL_tfce_vals = palm_tfce(REAL_corvals_map,tfce_opts,inmask);
REAL_tfce_map = zeros(sz_mask);
REAL_tfce_map(inmask) = REAL_tfce_vals;

fprintf('Input data: Volume size %i*%i*%i with %i masked voxels (%0.1f%% coverage) and %i subjects\n',...
    sz_mask(1),sz_mask(2),sz_mask(3),Nvoxels,mask_coverage_prc,Nsubs);
fprintf('Starting permutations with %i iterations and %i workers (%s)\n',cfg.iter,cfg.NumWorkers,datestr(now,'HH:MM:SS'))
pause(0.05); % pause to allow printing
tic;

% create null
exceedances_tfce_maxstat = zeros(1,Nvoxels);
exceedances_tfce = zeros(1,Nvoxels);
exceedances_raw = zeros(Nvoxels,1);
cluster_max_sizes = nan(cfg.iter,1);
parfor (iter = 1:cfg.iter,cfg.NumWorkers)
%for iter = 1:cfg.iter           
    % get permuted model
    [corvals,pvals] =compute_accuracy(data,cfg.target(perms_sets(:,iter)),model_params);
    
    corvals_map = zeros(sz_mask);
    corvals_map(inmask) = corvals;
    
    pvals_map = zeros(sz_mask);
    pvals_map(inmask) = 1-pvals; % higher = better
    
    cluster_max_sizes(iter) = palm_clustere_vol(pvals_map,1-cfg.p_val_threshold,inmask);
        
    % get voxel-wise (uncorrected stats)
    curexceeds = corvals >= REAL_corvals;
    exceedances_raw = exceedances_raw + curexceeds;
    
    % make tfce map
    tfce_vals = palm_tfce(corvals_map,tfce_opts,inmask);
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
results.raw_correlation_map = REAL_corvals_map;
results.raw_tfce_map = REAL_tfce_map;
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

fprintf('Voxels at corrected p<0.01: %i (voxel-wise FDR), %i (TFCE), %i (cluster extend)\n',...
    nnz(results.stats.raw_pval_corrected<0.01),...
    nnz(results.stats.tfce_pval_corrected<0.01),...
    nnz(results.stats.cluster_pval_corrected<0.01));

fprintf('---- All finished (%s) ----\n',datestr(now,'HH:MM:SS'));

end

function [corvals,pvals] =compute_accuracy(corMat,target,model_params)
% k nearest neighbors learner for regression and classification

x=size(corMat,1);
N=(0.5+sqrt(0.5^2+x*2));

K = model_params.k_val;
true_triu = triu(true(N),1);

results=zeros(size(corMat,2),N);
for sub=1:N
    
    subs=1:N;
    subs(sub)=[];
    grps=target;
    grps(sub)=[];
    mask=false(N);
    mask(sub,:)=true;
    mask(:,sub)=true;
    
    ind = mask(true_triu);
    [similarity,idx]=sort(corMat(ind,:),'descend');
    
    if model_params.isRegression==1,
        w = grps(idx(1:K,:));
        if model_params.isWeighted==1,
            % weight the neighbors based on their distance
            w0 = similarity(1:K,:);
            w0 = bsxfun(@times,w0,1./sum(w0,1));
            w = grps(idx(1:K,:)).*w0;
        else
            % uniform weighting
            w = grps(idx(1:K,:))/K;
        end
        results(:,sub)=results(:,sub) + sum(w',2);
    else
        results(:,sub)=results(:,sub) + mode(grps(idx(1:K,:))==target(sub))';
    end
end

if model_params.isRegression==1,
    [corvals,pvals] = corr(results',target,'Tail','right');
else
    corvals = sum(results,2);
    p = nnz(target==target(1))/N;
    p = max(p,1-p);
    pvals = binocdf(corvals,N,p,'upper');
    corvals = corvals/N;
end

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
