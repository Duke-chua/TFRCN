function miss = do_fast_rcnn_test(conf, model_stage, imdb, roidb, ignore_cache)
    if ~exist('ignore_cache', 'var')
        ignore_cache            = false;
    end

    [~,aboxes]                  = fast_rcnn_test(conf, imdb, roidb, ...
                                    'net_def_file',     model_stage.test_net_def_file, ...
                                    'net_file',         model_stage.output_model_file, ...
                                    'cache_name',       model_stage.cache_name, ...
                                    'ignore_cache',     ignore_cache);
    aboxes = aboxes{1};
    % for pedestrians detection
    
    miss = result_eval(conf,aboxes,model_stage,imdb,roidb,'fast_rcnn_cachedir');
end
