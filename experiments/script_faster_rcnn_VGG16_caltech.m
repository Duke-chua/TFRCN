function script_faster_rcnn_VGG16_caltech()
% script_faster_rcnn_VGG16_caltech()
% Faster rcnn training and testing with VGG16 model
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2017, Zhewei Xu
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
% opts.gpu_id                 = auto_select_gpu;
opts.gpu_id                 = 4;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

exp_name = 'faster_rcnn_caltech_VGG16';
% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_Faster_RCNN_caltech;
% cache base
cache_base_proposal         = 'faster_rcnn_caltech_VGG16';
cache_base_fast_rcnn        = '';
% train/test data
dataset                     = [];
use_flipped                 = false;
dataset                     = Dataset.caltech_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.caltech_test(dataset, 'test', use_flipped);

%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config(model);
conf_fast_rcnn              = fast_rcnn_config(model);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_caltech(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% proposal
conf_proposal.method_name   = 'stage1-rpn';
dataset.roidb_train         = cellfun(@(x, y) Faster_RCNN_Train.do_generate_proposal_caltech(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test          = Faster_RCNN_Train.do_generate_proposal_caltech(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
% test
model.stage1_rpn.nms        = model.final_test.nms;
[~,opt.stage1_rpn_miss]     = Faster_RCNN_Train.do_proposal_test_caltech(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
conf_fast_rcnn.exp_name     = exp_name;
model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_caltech(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% test
conf_fast_rcnn.method_name  = 'stage1-fast-rcnn';
[~,opts.stage1_fast_miss]   = Faster_RCNN_Train.do_fast_rcnn_test_caltech(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn            = Faster_RCNN_Train.do_proposal_train_caltech(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% proposal
conf_proposal.method_name   = 'stage2-rpn';
dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_generate_proposal_caltech(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test        	= Faster_RCNN_Train.do_generate_proposal_caltech(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% test
model.stage2_rpn.nms        = model.final_test.nms;
[~,opt.stage2_rpn_miss]     = Faster_RCNN_Train.do_proposal_test_caltech(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_caltech(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);

%% final test
fprintf('\n***************\nfinal test\n***************\n');

conf_proposal.method_name   = 'rpn';
model.stage2_rpn.nms        = model.final_test.nms;
% proposal
dataset.roidb_test          = Faster_RCNN_Train.do_generate_proposal_caltech(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% test
conf_fast_rcnn.method_name  = 'fast-rcnn';
model.stage2_fast_rcnn.nms  = model.final_test.nms;
[~,opts.final_fast_miss]    = Faster_RCNN_Train.do_fast_rcnn_test_caltech(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_caltech(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales',  2.^[3:5],...
                                    'exp_name', conf.exp_name);
end

function conf = proposal_config(model)
    conf = proposal_config_caltech('image_means', model.mean_image,...
                                   'feat_stride', model.feat_stride ...
                                  ,'scales',        480  ...
                                  ,'max_size',      640  ...
                                  ,'ims_per_batch', 1    ...
                                  ,'batch_size',    256  ...
                                  ,'fg_fraction',   0.5  ...
                                  ,'bg_weight',     1.0  ...
                                  ,'fg_thresh',     0.7  ...
                                  ,'bg_thresh_hi',  0.3  ...
                                  ,'bg_thresh_lo',  0    ...
                                  ,'test_scales',   480  ...
                                  ,'test_max_size', 640  ...
                                  ,'test_nms',      0.3  ...
                                   );
    
end

function conf = fast_rcnn_config(model)
    conf = fast_rcnn_config_caltech('image_means',   model.mean_image ...
                                   ,'scales',        480  ...
                                   ,'max_size',      640  ...
                                   ,'ims_per_batch', 2    ...
                                   ,'batch_size',    128  ...
                                   ,'fg_fraction',   0.25 ...
                                   ,'fg_thresh',     0.5 ...
                                   ,'bg_thresh_hi',  0.5  ...
                                   ,'bg_thresh_lo',  0.1  ...
                                   ,'bbox_thresh',   0.5  ...
                                   ,'test_scales',   480  ...
                                   ,'test_max_size', 640  ...
                                   ,'test_nms',      0.3  ...
                                    );
end