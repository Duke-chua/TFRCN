function script_test()
% script_test()
% --------------------------------------------------------
% RPN_BF
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
opts.gpu_id                 = 2;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% ouput
exp_name = 'VGG16_scut';
% exp_name = 'RPN_kv_kaist_lwir';
% exp_name = 'RPN_kv_kaist_lwir_flip';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_rpn_pedestrian_kaist_lwir('VGG16_caltech');
% cache base
cache_base_proposal         = 'rpn_kaist_lwir_vgg_16layers'; % output/cache_base_proposal
% train/test data
dataset                     = [];
use_flipped                 = false;
dataset                     = Dataset.scut_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.scut_test(dataset, 'test',false);

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_scut('image_means', model.mean_image, 'feat_stride', model.feat_stride);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, model);
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

%% read the RPN model
log_dir = fullfile(pwd, 'output', 'VGG16_kaist_lwir', 'rpn_cachedir', model.stage1_rpn.cache_name, 'train');
final_model_path = fullfile(log_dir, 'final');
if exist(final_model_path, 'file')
    model.stage1_rpn.output_model_file = final_model_path;
else
    error('RPN model does not exist.');
end
%% test
fprintf('\n***************\nstage one RPN test\n***************\n');
cache_name = 'RPN_scut_kaistrpn'; % output/exp_name/rpn_cahedir/cache_name
method_name = 'RPN-ped-kaist'; % external/piotr-toolbox-kaist/data-XXXX/res/method_name
Faster_RCNN_Train.do_proposal_test_scut(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, cache_name, method_name);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
% conf                  - struct rpn parameter
% cache_name            - [] rpn model name
% test_net_def_file     - [] address model prototxt file address
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_caltech(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales',  2.6*(1.3.^(0:8)), ... % anchors scales  
                                    'ratios', [1 / 0.41], ... % pedestrain high with ratio
                                    'exp_name', conf.exp_name);
end