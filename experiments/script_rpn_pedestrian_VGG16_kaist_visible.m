function script_rpn_pedestrian_VGG16_kaist_visible()
% script_rpn_pedestrian_VGG16_kaist_visible()
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2017, Zhewei Xu
% Licensed under TByrhe MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
% clear mex;
% clear is_valid_handle; % to clear init_key
% run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
% opts.gpu_id                 = auto_select_gpu;
opts.gpu_id                 = 1;
% active_caffe_mex(opts.gpu_id, opts.caffe_version);

exp_name = 'VGG16_kaist_visible';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_rpn_pedestrian_kaist_visible(exp_name);
% cache base
cache_base_proposal         = 'rpn_kaist_visible_vgg_16layers';
% train/test data
dataset                     = [];
% use_flipped                 = true;
% dataset                     = Dataset.caltech_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.kaist_visible_trainval(dataset, 'train');
% dataset                     = Dataset.caltech_test(dataset, 'test', false);
dataset                     = Dataset.kaist_visible_test(dataset, 'test');

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_kaist('image_means', model.mean_image, 'feat_stride', model.feat_stride);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder_kaist_visible(cache_base_proposal, model);
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

                        
%%  train
fprintf('\n***************\nstage one RPN train\n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_kaist_visible(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% model.stage1_rpn.output_model_file = '/mnt/RD/Code/ubuntu/RPN_BF_FIR/output/VGG16_kaist_visible/rpn_cachedir/rpn_kaist_visible_vgg_16layers_stage1_rpn/final';
%% test
fprintf('\n***************\nstage one RPN test\n***************\n');
cache_name = 'kaist_visible';
method_name = 'RPN-ped';
Faster_RCNN_Train.do_proposal_test_kaist_visible(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, cache_name, method_name);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
% conf                  - struct rpn parameter
% cache_name            - [] rpn model name
% test_net_def_file     - [] address model prototxt file address
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_caltech(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales',  2.6*(1.3.^(0:8)), ... % anchors scales [2.6000    3.3800    4.3940    5.7122    7.4259    9.6536   12.5497   16.3146   21.2090]
                                    'ratios', [1 / 0.41], ... % pedestrain high with ratio
                                    'exp_name', conf.exp_name);
end