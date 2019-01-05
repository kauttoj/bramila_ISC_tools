function results=bramila_mantel_ISC(cfg)
% results=bramila_mantel_ISC(cfg)
%
% Whole brain Mantel (aka RSA) test for ISC similarity matrices. Input 4D nifti file
% needs to contain the upper triangle elements of the similarity matrix
% (same format as the output of ISC toolbox with option "store correlation matrices"). 
% Statistics include standard FDR, TFCE and cluster extent.
%
% Input parameters:
%
%   cgf.infile= 4D volume or path to NIFTI, every volume (4th index) is a pair of subjects' similarity (MANDATORY)
%     NOTE: These are similarity values, i.e., high ISC = high similarity. This cannot be automatically checked!
%   cfg.mask = 3D brain mask or path to NIFTI, all values >0 are included (MANDATORY)
%   cfg.model = model dissimilarity matrix of interest, same size as ISC matrices (MANDATORY)
%   cfg.modelNI = cell vector of model matrices of NON-interest. Will be regressed out from cfg.model (OPTIONAL)
%   cfg.p_val_threshold = cluster forming threshold p-value in using parametric test (OPTIONAL)
%   cfg.NumWorkers = number of requested workers in parfor (default from the local profile) (OPTIONAL)
%   cfg.type = correlation type, "Pearson", "Spearman" or "Kendall" (default Spearman) (OPTIONAL)
%   cfg.iter = number of permutations (default 5000) (OPTIONAL)
%   cfg.doFisherTransform = do Fisher transform for ISC values (default=1, OPTIONAL), skip if already converted. No effect if correlation is other than Pearson!
%
% Output results struct with following fields:
%
%                      results.raw_tval_map: 3D voxel-wise correlation values
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

% if no input is given, create and analyze test data
if nargin==0,    
    N_subj = 15;    
    cfg.mask = zeros(41,45,47);
    cfg.mask(5:38,12:42,2:41)=1;
    N_pairs = N_subj*(N_subj-1)/2;
    cfg.infile= randn(41,45,47,N_pairs);
    inds = find(triu(ones(N_subj,N_subj),1));
    cfg.model = corr(randn(500,N_subj));    
    % create one big high-correlation cube
    total_significant = 0;
    for x=22:30
        for y=29:39
            for z=10:30
                total_significant=total_significant+1;
                nullmodel = corr(randn(500,N_subj));
                cfg.infile(x,y,z,:)=0.30*cfg.model(inds) + 0.70*nullmodel(inds);
            end
        end
    end    
    cfg.model = 1-cfg.model; % make into dissimilarity
    % single "wild voxel" with very high correlation, but no neighbors
    cfg.infile(7,13,9,:) = 0.50*cfg.model(inds) + 0.50*nullmodel(inds);    
    %cfg.modelNI = [];
    cfg.iter = 1000;    
    cfg.NumWorkers=4;
    RESULTS=bramila_mantel_ISC(cfg);  
    return
end

fprintf('---- Running "bramila_mantel_ISC" RSA (Mantel) test for volumetric data (%s) ----\n',datestr(now,'HH:MM:SS'));

% fix seed for repeatability
rng(666);

% p-val threshold for cluster extend correction
if ~isfield(cfg,'p_val_threshold')
   cfg.p_val_threshold = 0.01;
end
assert(cfg.p_val_threshold<=0.01); % do not allow threshold above 0.01

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

% initialize type
if(~isfield(cfg,'type'))
    cfg.type='Spearman';
end
assert(ismember(cfg.type,{'Pearson','Spearman','Kendall'}));

% initialize iter
if(~isfield(cfg,'iter'))
    cfg.iter=5000;
end
cfg.iter=max(10,cfg.iter);
assert(cfg.iter<50000); % too much will likely lead to memory issues

if cfg.iter<5000
   warning('!! Only %i iterations used, results are unreliable !!',cfg.iter);
end

%% starting
modelMat=cfg.model;
model_sz=size(modelMat);
Nsubs = model_sz(1);

if Nsubs<6
    error('Cannot continue with only %i subjects!',Nsubs)
end

% ISC matrix indices - THIS IS THE STANDARD FOR ISC TOOLBOX!
ISC_mat_inds=find(triu(ones(Nsubs),1));
NISCvalues = length(ISC_mat_inds);

% initialize modelNI to default
if(~isfield(cfg,'modelNI'))
    cfg.modelNI=0;
else
    assert(iscell(cfg.modelNI));
    assert(size(cfg.modelNI,1)==size(modelMat,1) && size(cfg.modelNI,1)==size(modelMat,1));
    if(size(cfg.modelNI,1)==size(modelMat))
        nuisancemats = [];
        for i=1:length(cfg.modelNI)
            nuisancemats = [nuisancemats,cfg.modelNI{i}(ISC_mat_inds)];
        end
        % cleaning the model
        % add if there's more than one models to regress
        [~,~,residu]=regress(modelMat(ISC_mat_inds),[nuisancemats,ones(size(ISC_mat_inds))]);
        tempmodelMat=zeros(size(modelMat));
        tempmodelMat(ISC_mat_inds)=residu/max(residu);
        tempmodelMat=tempmodelMat+tempmodelMat'+eye(size(tempmodelMat));
        modelMat=tempmodelMat;
    end
    clear tempmodelMat nuisancemats
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
    data = atanh(data);
    assert(numNans == nnz(isnan(data)),'Fisher transformation failed, NaN values emerged! Check your input data!');
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

% test that they are symmetrical matrices
temp=modelMat -modelMat';
if(sum(temp(:))~=0), error('Model matrix is not symmetrical'); end

% test that they both are similarity or dissimilarity matrices
if(sum(abs(diag(modelMat)))~=0), error('Model matrix is not a dissimilarity matrix ('); end

% test that they are square
if(model_sz(1) ~= model_sz(2)), error('model matrix is not square'); end
if( length(ISC_mat_inds) ~= size(data,4)), error('model matrix and brain data do not have the same size'); end

N_model_ranks = length(unique(modelMat(ISC_mat_inds)));
assert(N_model_ranks>1);
if N_model_ranks<10 && ~strcmp(cfg.type,'Kendall')
   warning('!! Your model seems categorical (%i unique values), consider using type ''Kendall'' !!',N_model_ranks) 
end

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
%!!!!!!!!!!!!!!!!!!!!
% convert similarities into distances, this is very important
data = 1-data;
%!!!!!!!!!!!!!!!!!!!!

% compute real, unpermuted values
[REAL_corvals,pvals] =corr(data,modelMat(ISC_mat_inds),'type',cfg.type,'tail','right');
if nnz(REAL_corvals==-1 | REAL_corvals==1)>0
   warning('!! Model is perfecty correlated with the data (%i voxels) !!',nnz(REAL_corvals==-1 || REAL_corvals==1)); 
end
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
    modelMat_null = modelMat(perms_sets(:,iter),perms_sets(:,iter));
    [corvals,pvals] =corr(data,modelMat_null(ISC_mat_inds),'type',cfg.type,'tail','right');
    
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
results.modelMat = modelMat;
results.modelMat_ind = ISC_mat_inds;
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
