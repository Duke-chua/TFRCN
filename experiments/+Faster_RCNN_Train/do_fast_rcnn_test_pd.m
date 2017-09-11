function [aboxes,miss] = do_fast_rcnn_test_pd(conf, model_stage, imdb, roidb, ignore_cache)
    if ~exist('ignore_cache', 'var')
        ignore_cache            = false;
    end

    aboxes                      = fast_rcnn_test_pd(conf, imdb, roidb, ...
                                    'net_def_file',     model_stage.test_net_def_file, ...
                                    'net_file',         model_stage.output_model_file, ...
                                    'cache_name',       model_stage.cache_name, ...
                                    'ignore_cache',     ignore_cache, ...
                                    'exp_name',         conf.exp_name);
    %% for catlech only one class
    aboxes = aboxes{1};
    diary on;
    %% nms
    fprintf('Doing nms ... \n'); 
    aboxes                      = boxes_filter(aboxes, -1, 0.5, -1, conf.use_gpu);
    aboxes                      = boxes_thres(aboxes, 40);
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
    
    %% eval the roc
    fprintf('Preparing the results for Caltech evaluation ...');
    cache_dir = fullfile(pwd, 'output', conf.exp_name, 'fast_rcnn_cachedir', model_stage.cache_name);
    res_boxes = aboxes;
    mkdir_if_missing(fullfile(cache_dir, conf.method_name));
    % remove all the former results
    DIRS=dir(fullfile(fullfile(cache_dir, conf.method_name))); 
    n=length(DIRS);
    for i=1:n
        if (DIRS(i).isdir && ~strcmp(DIRS(i).name,'.') && ~strcmp(DIRS(i).name,'..') ) % except . ..
            rmdir(fullfile(cache_dir, conf.method_name ,DIRS(i).name),'s'); % remove include subdir
        end
    end

    assert(length(imdb.image_ids) == size(res_boxes, 1));
    bbsNm = fullfile(cache_dir, [conf.method_name '_Det.txt']);
    if(exist(bbsNm,'file')), delete(bbsNm); end
    fid2 = fopen(bbsNm, 'w');
    for i = 1:size(res_boxes, 1)
        if ~isempty(res_boxes{i})
            sstr = strsplit(imdb.image_ids{i}, '_');
            mkdir_if_missing(fullfile(cache_dir, conf.method_name, sstr{1}));
            fid = fopen(fullfile(cache_dir, conf.method_name, sstr{1}, [sstr{2} '.txt']), 'a');
            % transform [x1 y1 x2 y2] to [x y w h], for matching the
            % caltech evaluation protocol
            res_boxes{i}(:, 3) = res_boxes{i}(:, 3) - res_boxes{i}(:, 1); % h
            res_boxes{i}(:, 4) = res_boxes{i}(:, 4) - res_boxes{i}(:, 2); % w
            for j = 1:size(res_boxes{i}, 1)
                fprintf(fid, '%d,%f,%f,%f,%f,%f\n', str2double(sstr{3}(2:end))+1, res_boxes{i}(j, :)); % dirs result
                fprintf(fid2, '%d,%.2f,%.2f,%.2f,%.2f,%.2f\n', i,  res_boxes{i}(j, :)); % one file
            end
            fclose(fid);
        end
    end
    fclose(fid2);
    fprintf('Done.');
    
    % run evaluation using bbGt
    [gt,dt] = bbGt('loadAll',roidb.anno_path,bbsNm,conf.eval_pLoad);
    [gt,dt] = bbGt('evalRes',gt,dt,conf.bbox_thresh,conf.eval_mul);
    [fp,tp,score,miss] = bbGt('compRoc',gt,dt,1,conf.eval_ref);
    miss=exp(mean(log(max(1e-10,1-miss)))); 
    fprintf('miss rate:%.2f\n', miss*100);

    % optionally plot roc
    show = 0;
    if(show)
    figure(show); 
    plotRoc([fp tp],'logx',1,'logy',1,'xLbl','fppi',...
    'lims',[3.1e-3 1e1 .05 1],'color','g','smooth',1,'fpTarget',conf.eval_ref);
    title(sprintf('log-average miss rate = %.2f%%',miss*100));
    savefig([fullfile(cache_dir, conf.method_name) 'Roc'],show,'png');
    end

    % copy results to eval folder and run eval script to get figure.
    folder1 = fullfile(pwd, 'output', conf.exp_name, 'fast_rcnn_cachedir', model_stage.cache_name, conf.method_name);
    folder2 = fullfile(pwd, 'external', 'code3.2.1', 'data-USA', 'res', conf.method_name);
    mkdir_if_missing(folder2);
    copyfile(folder1, folder2);
    if(0)
        tmp_dir = pwd;
        cd(fullfile(pwd, 'external', 'code3.2.1'));
        dbEval_RPNBF;
        cd(tmp_dir);
    end

    diary off;
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
% do nms
    % to speed up nms
    if per_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), per_nms_topN), :), aboxes, 'UniformOutput', false); % make sure box not excced, get 1:min(size(x, 1), per_nms_topN) aboxes
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        if use_gpu
            for i = 1:length(aboxes)
                tic_toc_print('nms: %d / %d \n', i, length(aboxes));
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :); % do nms
            end
        else
            parfor i = 1:length(aboxes)
                aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
            end
        end
    end
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        aboxes = cellfun(@(x) x(1:min(size(x, 1), after_nms_topN), :), aboxes, 'UniformOutput', false); % only keep after_nms_topN boxes
    end
end

function aboxes = boxes_thres(aboxes, max_per_image)
    num_images = length(aboxes);
    %heuristic: keep an average of 40 detections per class per images prior to NMS
    top_k = max_per_image * num_images;
    
    % Keep top K
    X = cat(1, aboxes{:});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for i = 1:num_images
        aboxes{i} = aboxes{i}(aboxes{i}(:,end) > thresh, :);
    end
end
