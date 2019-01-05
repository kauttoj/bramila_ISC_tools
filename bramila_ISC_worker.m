function bramila_ISC_worker(cfg_file)
% bramila_ISC_worker(cfg_file)
% worker function called by bramila_ISC function
% run in one of the two modes: mode 0 for making dataset, mode 1 for unpermuted data (fast) and mode 2 with permutations (slow)

fprintf('Loading cfg file %s\n',cfg_file);

load(cfg_file);

fprintf('stage variable is %i\n',stage);

if stage == 0 % mode 0
    
    fprintf('Starting ''bramila_ISC_worker'' with MODE = %i (%s)\n',stage,datestr(now,'HH:MM:SS'));  
    addpath(cfg.niftitools_path);
    group = cfg.group_id(cfg.process_index);
    file = cfg.infiles{cfg.process_index};    
    fprintf('..loading and saving data for subject %i of %i (group %i): %s (%s)\n',cfg.process_index,cfg.Nsubs,group,file,datestr(now,'HH:MM:SS'));
    nii=load_nii(file);
    data = nii.img;
    for i=1:3
        assert(cfg.sz_mask(i)==size(data,i));
    end
    data = reshape(data,[],size(data,4));
    data = single(data(cfg.mask_ind,:)); % need to do this now to avoid numeric inaccuracy    
    data = zscore(data,[],2);
          
    % https://www.mathworks.com/matlabcentral/answers/96360-why-do-the-mean-and-std-functions-return-the-wrong-result-for-large-single-precision-matrices-in-any
    % trick to force mean close to zero
    data=data-mean(data,2);
    data=data-mean(data,2);
    
    varvec = var(data,[],2);
    bad_voxels = find(varvec<0.99 | isnan(varvec));
    if ~isempty(bad_voxels)
       warning('Bad data found in %i voxels!!!',length(bad_voxels));
    end    

    sz_data = size(data);
    
    varvec(bad_voxels)=[];
    m = abs(varvec-1);
    [x,y,z]=ind2sub(sz_data,cfg.mask_ind(find(m==max(m),1,'first')));
    fprintf('maximum deviation from unity variance was %f\n',max(m));
    assert(max(m)<1e-5,sprintf('Failed to standardize variance! Check voxel (%i,%i,%i)!',x,y,z));       

    m=abs(mean(data,2));
    m(bad_voxels)=[];
    [x,y,z]=ind2sub(sz_data,cfg.mask_ind(find(m==max(m),1,'first')));
    fprintf('maximum deviation from zero mean was %f\n',max(m));
    assert(max(m)<1e-5,sprintf('Failed to standardize mean! Check voxel (%i,%i,%i)!',x,y,z));
    
    % save data
    mask = cfg.mask;
    save(cfg.stage0_resultfiles{cfg.process_index},'data','file','mask','sz_data','bad_voxels','-v7.3');
    
    % update stage of current file
    stage = 1;
    save(cfg_file,'stage','-append');
    
else
    
    Nvoxels=cfg.Nvoxel;
    Nsubs = cfg.Nsubj1 + cfg.Nsubj2;
    Ntime = cfg.Ntime;
    sz_mask=cfg.sz_mask;
    inmask = cfg.mask_ind;
    tfce_opts=cfg.tfce_opts;        
    
    fprintf('Starting ''bramila_ISC_worker'' for %i voxels, %i timepoints, %i+%i subjects with MODE = %i (%s)\n',Nvoxels,Ntime,cfg.Nsubj1,cfg.Nsubj2,stage,datestr(now,'HH:MM:SS'));
    
    all_segments = round(linspace(1,Nvoxels+1,cfg.data_split_parts+1));
    
    if stage == 1 % mode 1                
        
        REAL_tvals=[];
        pvals=[];
        for segment_nr = 1:cfg.data_split_parts
            segment = all_segments(segment_nr):(all_segments(segment_nr+1)-1);
            fprintf('...loading data segment %i of %i\n',segment_nr,cfg.data_split_parts);
            alldata = load_data(cfg,Nsubs,segment);                
            fprintf('...computing unpermuted values\n');
            [~,REAL_tvals0,pvals0] = permute_and_compute(alldata,0,cfg.perm_method,cfg.doFisherTransform,cfg.group1_ind,cfg.group2_ind);
            REAL_tvals=[REAL_tvals;REAL_tvals0];
            pvals=[pvals;pvals0];
            clear alldata;            
        end
        
        assert(nnz(isnan(REAL_tvals))==0 && nnz(~isfinite(REAL_tvals))==0,'illegal tvals found!');
        
        fprintf('..computing maps\n');
        
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
        
        fprintf('..saving results into %s\n',cfg.stage1_resultfiles);
        
        save(cfg.stage1_resultfiles,...
            'REAL_tvals',...
            'REAL_tvals_map',...
            'REAL_clstat',...
            'REAL_clstat_map',...
            'REAL_tfce_vals_signed',...
            'REAL_tfce_vals',...
            'REAL_tfce_map','-v7.3');
        
        stage = 2;
        save(cfg_file,'stage','-append');
        
    elseif stage == 2 % mode 2
        
        % set unique seed to make sure we are not repeating same permutations between workers
        rng(cfg.process_index);
        
        % load unpermuted data to compare against
        load(cfg.stage1_resultfiles);
        
        % create null arrays
        exceedances_tfce_maxstat = zeros(Nvoxels,1);
        exceedances_tfce = zeros(Nvoxels,1);
        exceedances_raw = zeros(Nvoxels,1);
        exceedances_raw_maxstat = zeros(Nvoxels,1);
        cluster_max_sizes = nan(cfg.iter,1);
        
        % initialize
        failed_status = 0;
        REAL_tvals_abs = abs(REAL_tvals);
        
        fprintf('..starting permutations with %i iterations\n',cfg.iter);
        
        perm_rate = nan;
        tic();
        
        for iter = 1:cfg.iter
                        
            if cfg.data_split_parts>1 % deal with splitting
                if iter==1 % compute all permuted maps first
                    allshifts=[];
                    alltvals=nan(Nvoxels,cfg.iter);
                    allpvals=nan(Nvoxels,cfg.iter);
                    for segment_nr = 1:cfg.data_split_parts
                        segment = all_segments(segment_nr):(all_segments(segment_nr+1)-1);
                        fprintf('...loading data segment %i of %i\n',segment_nr,cfg.data_split_parts);
                        alldata = load_data(cfg,Nsubs,segment);  
                        fprintf('...computing permuted values for all %i iterations\n',cfg.iter);
                        for iter0 = 1:cfg.iter 
                            if segment_nr==1, % collect shifts
                                [alldata,tvals0,pvals0,shifts] = permute_and_compute(alldata,1,cfg.perm_method,cfg.doFisherTransform,cfg.group1_ind,cfg.group2_ind);
                                allshifts=[allshifts;shifts]; % store shifts
                            else % apply stored shifts
                                [alldata,tvals0,pvals0] = permute_and_compute(alldata,1,cfg.perm_method,cfg.doFisherTransform,cfg.group1_ind,cfg.group2_ind,allshifts(iter0,:));
                            end                            
                            alltvals(segment,iter0)=tvals0;
                            allpvals(segment,iter0)=pvals0;
                        end
                        clear alldata tvals0 pvals0;
                    end
                    clear allshifts
                    tvals=alltvals(:,iter);
                    pvals=allpvals(:,iter);                    
                else % just load maps
                    fprintf('...using precomputed values\n');
                    tvals=alltvals(:,iter);
                    pvals=allpvals(:,iter);
                end
            else
                % no splits, life is simple
                fprintf('..computing permuted values (full segment)\n');
                if iter==1 % load data once
                    alldata = load_data(cfg,Nsubs,1:Nvoxels);
                end
                [alldata,tvals,pvals] = permute_and_compute(alldata,1,cfg.perm_method,cfg.doFisherTransform,cfg.group1_ind,cfg.group2_ind);
            end
            
            if not( nnz(isnan(tvals))==0 && nnz(~isfinite(tvals))==0 )
                fprintf('..FOUND ILLEGAL VALUES IN PERMUTED RESULTS, TERMINATING!!');
                failed_status = 1;
                break;
            end
            
            tvals_map = zeros(sz_mask);
            tvals_map(inmask) = tvals;
            
            pvals_map = zeros(sz_mask);
            pvals_map(inmask) = 1-pvals; % higher = better
            
            % get maximum cluster of both tails (makes no sense to merge positive and negative)
            cluster_max_sizes(iter) = max(...
                palm_clustere_vol(pvals_map.*(tvals_map>0),1-cfg.p_val_threshold,inmask),...
                palm_clustere_vol(pvals_map.*(tvals_map<0),1-cfg.p_val_threshold,inmask));
            
            % get voxel-wise (uncorrected stats)
            curexceeds = abs(tvals) >= REAL_tvals_abs;
            exceedances_raw = exceedances_raw + curexceeds;
            
            % get voxel-wise (FWE corrected stats)
            curexceeds = max(abs(tvals)) >= REAL_tvals_abs;
            exceedances_raw_maxstat = exceedances_raw_maxstat + curexceeds;
            
            % make tfce map (all positive!)
            tfce_vals = palm_tfce(tvals_map.*(tvals_map>0),tfce_opts,inmask) + palm_tfce(-tvals_map.*(tvals_map<0),tfce_opts,inmask);
            % get max stats
            curexceeds = max(tfce_vals) >= REAL_tfce_vals;
            exceedances_tfce_maxstat = exceedances_tfce_maxstat + curexceeds;
            % get voxel-wise (uncorrected stats)
            curexceeds = tfce_vals >= REAL_tfce_vals;
            exceedances_tfce = exceedances_tfce + curexceeds;
            
            perm_rate = toc()/iter;
            fprintf('..permutation %i done (rate 1 perm = %.1fs)\n',iter,perm_rate);
            
        end
        assert(nnz(isnan(cluster_max_sizes))==0);
        
        fprintf('..saving results into %s\n',cfg.stage2_resultfiles{cfg.process_index});
        
        save(cfg.stage2_resultfiles{cfg.process_index},...
            'exceedances_tfce_maxstat',...
            'exceedances_tfce',...
            'exceedances_raw',...
            'exceedances_raw_maxstat',...
            'cluster_max_sizes','failed_status','perm_rate','-v7.3')
        
        stage = 3;
        save(cfg_file,'stage','-append');
        
    else
        error('Stage must be 1 (unpermuted) or 2 (permuted)');
    end
end

fprintf('all done! (%s)\n',datestr(now,'HH:MM:SS'));

end

function alldata = load_data(cfg,Nsubs,this_segment)
    % store data into cell (does not require continuous memory block)
    % data is voxels x time
    alldata = cell(Nsubs,1);
    for i=1:Nsubs
        f = matfile(cfg.stage0_resultfiles{i});
        data = f.data(this_segment,:); % only take voxels of interest
        alldata{i}=data;
        % make sure that data is z-scored (only first voxel, assume holds for all)
        assert(abs(mean(data(1,:)))<1e-4,sprintf('data mean not 0 (%s)!',cfg.stage0_resultfiles{i}));
        assert(abs(var(data(1,:))-1)<1e-4,sprintf('data variance not 1 (%s)!',cfg.stage0_resultfiles{i}));
    end
end

function [alldata,tvals,pvals,shifts] = permute_and_compute(alldata,doPermute,perm_type,doFisherTransform,group1_ind,group2_ind,shifts)

if nargin<7
    shifts=[];
end

Nsub = length(alldata);
Nsub1 = length(group1_ind);
Nsub2 = length(group2_ind);

Nvoxel = size(alldata{1},1);
Ntime = size(alldata{1},2);

% permute data with chosen method
% null asumption: Stimulus is not time-locked
% Note: There are total (Ntime-1)^(Nsub-1) combinations of permutations
if doPermute
    if strcmp(perm_type,'circshift')
        if isempty(shifts)
            shifts = randi(Ntime-1,1,Nsub);
        end
        for i=1:Nsub            
            alldata{i} = circshift(alldata{i},shifts(i),2);
        end
    elseif strcmp(perm_type,'phasemix')
        error('not yet implemented!')
    end
end

m = 0;
for i=1:Nsub1
    m = m + alldata{group1_ind(i)};
end
corvals=zeros(Nvoxel,Nsub1);
for i=1:Nsub1 % iterate over each subject      
    m0=m-alldata{group1_ind(i)};
    m0=zscore(m0,[],2); % z-score averaged signals
    corvals(:,i)=sum(alldata{group1_ind(i)}.*m0,2);
end
corvals = corvals/(Ntime-1); % equivalent with corr2
corvals(corvals<-0.99999)=-0.99999;
corvals(corvals>0.99999)=0.99999;
if doFisherTransform
    corvals = atanh(corvals);
end

if Nsub2>1
    % there are two groups, repeat above for the second group
    m = 0;
    for i=1:Nsub2
        m = m + alldata{group2_ind(i)};
    end    
    corvals2=zeros(Nvoxel,Nsub2);
    for i=1:Nsub2
        m0=m-alldata{group2_ind(i)};
        m0=zscore(m0,[],2); % z-score averaged signals
        corvals2(:,i)=sum(alldata{group2_ind(i)}.*m0,2);
    end
    corvals2 = corvals2/(Ntime-1);
    corvals2(corvals2<-0.99999)=-0.99999;
    corvals2(corvals2>0.99999)=0.99999;
    if doFisherTransform
        corvals2 = atanh(corvals2);
    end
    
    % two sample t-vals
    [~,pvals,~,stats] = ttest2(corvals,corvals2,'Tail','both','Alpha',0.05,'Vartype','unequal','Dim',2);
    tvals = stats.tstat(:);
    pvals=pvals(:);
    
else

    % one sample t-vals
    [~,pvals,~,stats] = ttest(corvals,0.05,'Tail','right','Alpha',0.05,'Dim',2);
    tvals = max(0,stats.tstat(:)); % omit negative values
    pvals=pvals(:);
    
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
tfcestat = tfcestat(:) * dh;

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
