function script_faster_rcnn_VGG16_kaist_M1()
% script_faster_rcnn_VGG16_scut()
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

exp_name = 'faster_rcnn_kaist_VGG16_1';
% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_Faster_RCNN_hole; %M1
% cache base
cache_base_proposal         = 'vgg16-hole';
cache_base_fast_rcnn        = '';
% train/test data
dataset                     = [];
use_flipped                 = false;
dataset                     = Dataset.kaist_lwir_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.kaist_lwir_test(dataset, 'test', use_flipped);

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
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_pd(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% proposal
conf_proposal.method_name   = 'stage1-rpn-1-1';
dataset.roidb_train         = cellfun(@(x, y) Faster_RCNN_Train.do_generate_proposal_pd(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test          = Faster_RCNN_Train.do_generate_proposal_pd(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
% test
model.stage1_rpn.nms        = model.nms.test; %M1
[~,opt.stage1_rpn_miss]     = Faster_RCNN_Train.do_proposal_test_pd(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
conf_fast_rcnn.exp_name     = exp_name;
model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_pd(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% test
conf_fast_rcnn.method_name  = 'stage1-fast-rcnn-1-1';
[~,opts.stage1_fast_miss]   = Faster_RCNN_Train.do_fast_rcnn_test_pd(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn            = Faster_RCNN_Train.do_proposal_train_pd(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% proposal
conf_proposal.method_name   = 'stage2-rpn-1-1';
dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_generate_proposal_pd(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test        	= Faster_RCNN_Train.do_generate_proposal_pd(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% test
model.stage2_rpn.nms        = model.nms.test; %M1
[~,opt.stage2_rpn_miss]     = Faster_RCNN_Train.do_proposal_test_pd(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train_pd(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);

%% final test
fprintf('\n***************\nfinal test\n***************\n');

% proposal
model.stage2_rpn.nms        = model.nms.test_propsal; %M1
dataset.roidb_test          = Faster_RCNN_Train.do_generate_proposal_pd(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% test
conf_fast_rcnn.method_name  = 'fast-rcnn-1-1';
[~,opts.final_fast_miss]    = Faster_RCNN_Train.do_fast_rcnn_test_pd(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_pd(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_pd(cache_name, ...
                                    'scales',  2.^[3:5],...
                                    'exp_name', conf.exp_name);
end

function conf = proposal_config(model)
    conf = proposal_config_pd('image_means', model.mean_image,...
                              'feat_stride', model.feat_stride ...
                             ,'scales',        512  ...
                             ,'max_size',      640  ...
                             ,'ims_per_batch', 1    ...
                             ,'batch_size',    120  ...
                             ,'fg_fraction',   1/6  ...
                             ,'bg_weight',     1.0  ...
                             ,'fg_thresh',     0.5  ...
                             ,'bg_thresh_hi',  0.5  ...
                             ,'bg_thresh_lo',  0    ...
                             ,'test_scales',   512  ...
                             ,'test_max_size', 640  ...
                             ,'test_nms',      0.5  ...
                             ,'test_min_box_height',50 ...
                             ,'datasets',     'kaist' ...
                              );
    % for eval_pLoad
    pLoad = {'lbls',{'person'},'ilbls',{'people','person?','cyclist'},'squarify',{3,.41}};
    pLoad = [pLoad 'hRng',[55 inf], 'vType',{{'none','partial'}},'xRng',[5 635],'yRng',[5 475]];
    conf.eval_pLoad = pLoad;
end

function conf = fast_rcnn_config(model)
    conf = fast_rcnn_config_pd('image_means',   model.mean_image ...
                              ,'scales',        512  ...
                              ,'max_size',      640  ...
                              ,'ims_per_batch', 2    ...
                              ,'batch_size',    128  ...
                              ,'fg_fraction',   0.25 ...
                              ,'fg_thresh',     0.5  ...
                              ,'bg_thresh_hi',  0.5  ...
                              ,'bg_thresh_lo',  0.1  ...
                              ,'bbox_thresh',   0.5  ...
                              ,'test_scales',   512  ...
                              ,'test_max_size', 640  ...
                              ,'datasets',     'kaist' ...
                               );
    pLoad = {'lbls',{'person'},'ilbls',{'people','person?','cyclist'},'squarify',{3,.41}};
    pLoad = [pLoad 'hRng',[55 inf], 'vType',{{'none','partial'}},'xRng',[5 635],'yRng',[5 475]];
    conf.eval_pLoad = pLoad;
end