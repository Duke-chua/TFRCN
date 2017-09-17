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
    % for catlech only one class
    aboxes = aboxes{1};
    
    %% Evaluation
    miss = result_eval(conf, aboxes,model_stage,imdb,roidb);
    
    if(0)
        tmp_dir = pwd;
        cd(fullfile(pwd, 'external', 'code3.2.1'));
        dbEval_RPNBF;
        cd(tmp_dir);
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
