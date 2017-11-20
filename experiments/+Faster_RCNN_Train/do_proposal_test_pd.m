function [aboxes,miss] = do_proposal_test_pd(conf, model_stage, imdb, roidb)
    aboxes                      = proposal_test_pd(conf, imdb, ...
                                        'net_def_file',     model_stage.test_net_def_file, ...
                                        'net_file',         model_stage.output_model_file, ...
                                        'cache_name',       model_stage.cache_name); 
          
    %% Evaluation
    miss = result_eval(conf,aboxes,model_stage,imdb,roidb,'rpn_cachedir');

    if(0)
        tmp_dir = pwd;
        cd(fullfile(pwd, 'external', 'code3.2.1'));
        dbEval_RPNBF;
        cd(tmp_dir);
    end
end