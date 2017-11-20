function [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb_pd(conf, imdbs, roidbs, bbox_means, bbox_stds)
% [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb_pd(conf, imdbs, roidbs, cache_img, bbox_means, bbox_stds)
%   Gather useful information from imdb and roidb
%   pre-calculate mean (bbox_means) and std (bbox_stds) of the regression
%   term for normalization
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2017, Zhewei Xu
% Licensed under The MIT License [see LICENSE for details]
% -------------------------------------------------------- 
    
    if ~exist('bbox_means', 'var')
        bbox_means = [];
        bbox_stds = [];
    end
    
    if ~iscell(imdbs)
        imdbs = {imdbs};
        roidbs = {roidbs};
    end

    imdbs = imdbs(:);
    roidbs = roidbs(:);
    
    image_roidb = ...
        cellfun(@(x, y) ... // @(imdbs, roidbs)
                arrayfun(@(z) ... //@([1:length(x.image_ids)])
                        struct('image_path', x.image_at(z), 'image_id', x.image_ids{z}, 'im_size', x.sizes(z, :), 'imdb_name', x.name, ...
                        'overlap', y.rois(z).overlap, 'boxes', y.rois(z).boxes, 'gt_ignores', y.rois(z).ignores, 'class', y.rois(z).class, 'image', [], 'bbox_targets', []), ...
                [1:length(x.image_ids)]', 'UniformOutput', true),...
        imdbs, roidbs, 'UniformOutput', false);
    
    image_roidb = cat(1, image_roidb{:});

    % skip some empty image to save memory
    % empty_image_sample_step = 1 equals use each img
    empty_image_sample_step=1;
    if empty_image_sample_step > 0
        num_images = length(image_roidb);
        keep_idxes = zeros(num_images, 1);        
        keep_idxes(1:empty_image_sample_step:length(keep_idxes)) = 1;
        % add the non-empty image
        image_roidb_cell = num2cell(image_roidb, 2);
        nonempty = 0;
        for i = 1:num_images
            %if ~isempty(image_roidb_cell{i}.boxes)
            if ~isempty(find(image_roidb_cell{i}.gt_ignores == 0)) % check the gt_ignores
                keep_idxes(i) = 1;
                nonempty = nonempty+1;
            end
        end
        clear image_roidb_cell;
        fprintf('total (%d) = nonempty (%d) + empty (%d)\n', sum(keep_idxes), nonempty, sum(keep_idxes)-nonempty);
        image_roidb = image_roidb(logical(keep_idxes));
    end
    
    
    % enhance roidb to contain bounding-box regression targets
    [image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);

    % check 
    for i = 1:length(image_roidb)
        if ~isempty(find(image_roidb(i).gt_ignores < 1))
            if (sum(image_roidb(i).overlap>=0.5)==0)
                disp(['all ols < 0.5 warning: ' image_roidb(i).image_id, ' max ol = ', num2str(max(image_roidb(i).overlaps{1}))]);
            end
        end   
    end
end

function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
    % means and stds -- (k+1) * 4, include background class

    num_images = length(image_roidb);
    % Infer number of classes from the number of columns in gt_overlaps
    num_classes = size(image_roidb(1).overlap, 2);
    valid_imgs = true(num_images, 1);
    for i = 1:num_images
       rois = image_roidb(i).boxes;
       [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
           compute_targets(conf, rois, image_roidb(i).overlap, image_roidb(i).gt_ignores);
    end
    if ~all(valid_imgs)
        image_roidb = image_roidb(valid_imgs);
        num_images = length(image_roidb);
        fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
    end
        
    if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
        % Compute values needed for means and stds
        % var(x) = E(x^2) - E(x)^2
        class_counts = zeros(num_classes, 1) + eps;
        sums = zeros(num_classes, 4);
        squared_sums = zeros(num_classes, 4);
        for i = 1:num_images
           targets = image_roidb(i).bbox_targets;
           for cls = 1:num_classes
              cls_inds = find(targets(:, 1) == cls);
              if ~isempty(cls_inds)
                 class_counts(cls) = class_counts(cls) + length(cls_inds); 
                 sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
                 squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
              end
           end
        end

        means = bsxfun(@rdivide, sums, class_counts);
        stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
        
        % add background class
        means = [0, 0, 0, 0; means]; 
        stds = [0, 0, 0, 0; stds];
    end
    
    % Normalize targets
    for i = 1:num_images
        targets = image_roidb(i).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
                image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
            end
        end
    end
end


function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap, gt_ignores)

    overlap = full(overlap);

    [max_overlaps, max_labels] = max(overlap, [], 2);

    % Indices of ground-truth ROIs
    gt_inds_full = find(max_overlaps == 1);
    gt_inds = gt_inds_full(gt_ignores~=1, :);

    % ensure ROIs are floats
    rois = single(rois);
    
    bbox_targets = zeros(size(rois, 1), 5, 'single');
    
    if ~isempty(gt_inds)
        % Indices of examples for which we try to make predictions
        ex_inds = find(max_overlaps >= conf.bbox_thresh & max_overlaps < 1);

        % Get IoU overlap between each ex ROI and gt ROI
        ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));

        assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));

        % Find which gt ROI each ex ROI has max overlap with:
        % this will be the ex ROI's gt target
        [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
        gt_rois = rois(gt_inds(gt_assignment), :);
        ex_rois = rois(ex_inds, :);

        [regression_label] = fast_rcnn_bbox_transform(ex_rois, gt_rois);

        bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
        bbox_targets(gt_inds, 1) = 1;
    end
    
    % Select foreground ROIs as those with >= fg_thresh overlap
    is_fg = max_overlaps >= conf.fg_thresh;
    % ignores gt are not fg
    is_fg(gt_inds_full(gt_ignores==1)) = false;
    % Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
    is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;
    
    % check if there is any fg or bg sample. If no, filter out this image
    is_valid = true;
    if ~any(is_fg | is_bg)
        is_valid = false;
    end
end