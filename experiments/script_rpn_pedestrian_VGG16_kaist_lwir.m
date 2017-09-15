function script_rpn_pedestrian_VGG16_kaist_lwir()
% script_rpn_pedestrian_VGG16_kaist_lwir()
% --------------------------------------------------------
% RPN
% Copyright (c) 2017, Zhewei Xu
% Licensed under TByrhe MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
% opts.gpu_id                 = auto_select_gpu;
opts.gpu_id                 = 1;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% ouput
exp_name = 'VGG16_kaist_lwir';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_rpn_pedestrian_kaist_lwir('VGG16_caltech');

% cache base
cache_base_proposal         = 'rpn_kaist_lwir_vgg_16layers'; % output/cache_base_proposal
% train/test data
dataset                     = [];
use_flipped                 = false;
dataset                     = Dataset.kaist_lwir_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.kaist_lwir_test(dataset, 'test',false);

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config(model);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder_pd(cache_base_proposal, model);
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
                        
%%  train
fprintf('\n***************\nstage one RPN train\n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_pd(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

%% test
fprintf('\n***************\nstage one RPN test\n***************\n');
conf_proposal.method_name = 'RPN-ped'; % external/piotr-toolbox-kaist/data-XXXX/res/method_name
Faster_RCNN_Train.do_proposal_test_pd(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_pd(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_pd(cache_name, ...
                                    'scales',  2.6*(1.3.^(0:8)), ... % anchors scales  
                                    'ratios', [1 / 0.41], ... % pedestrain high with ratio
                                    'exp_name', conf.exp_name);
end

function conf = proposal_config(model)
    conf = proposal_config_pd('image_means', model.mean_image,...
                              'feat_stride', model.feat_stride ...
                             ,'scales',        480  ...
                             ,'max_size',      640  ...
                             ,'test_scales',   480  ...
                             ,'test_max_size', 640  ...
                             ,'datasets',     'kaist' ...
                              );
    % for eval_pLoad
    pLoad={'lbls',{'person'},'ilbls',{'people','person?','cyclist'},'squarify',{3,.41}}; % copy from acfDemoKAIST.m and add squarify see bbApply.m
    pLoad = [pLoad 'hRng',[55 inf], 'vType', {'none','partial'},'xRng',[5 635],'yRng',[5 475]]; % copy from acfDemoKAIST.m
    conf.eval_pLoad = pLoad;   
end