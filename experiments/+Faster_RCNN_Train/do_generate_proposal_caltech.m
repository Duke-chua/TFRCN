function roidb_new = do_generate_proposal_caltech(conf, model_stage, imdb, roidb)
      
    aboxes                      = proposal_test_caltech(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name); 
    diary on;
    %% nms
    fprintf('Doing nms ... ');                               
    aboxes                      = boxes_filter(aboxes, model_stage.nms.per_nms_topN, model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);

    roidb_regions               = make_roidb_regions(aboxes, imdb.image_ids);

    roidb_new                   = roidb_from_proposal_score(imdb, roidb, roidb_regions, ...
                                    'keep_raw_proposal', false);
    %% eval the gt recall
    gt_num = 0;
    gt_re_num = 0;
    for i = 1:length(roidb.rois)
        gts = roidb.rois(i).boxes(roidb.rois(i).ignores~=1, :); % keep not ignores gt
        if ~isempty(gts)
            rois = aboxes{i}(:, 1:4); % proposal roidb
            max_ols = max(boxoverlap(rois, gts)); % compute IoU
            gt_num = gt_num + size(gts, 1); % count gt num
            gt_re_num = gt_re_num + sum(max_ols >= 0.5); % count recall gt num
        end
    end
    fprintf('gt recall rate = %.4f\n', gt_re_num / gt_num);

    diary off;
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), per_nms_topN), :), aboxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1  
        if use_gpu
            for i = 1:length(aboxes)
                tic_toc_print('nms: %d / %d \n', i, length(aboxes));
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
            end
        else
            parfor i = 1:length(aboxes)
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
            end
        end
    end
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), after_nms_topN), :), aboxes, 'UniformOutput', false);
    end
end

function regions = make_roidb_regions(aboxes, images)
    regions.boxes = aboxes;
    regions.images = images;
end
