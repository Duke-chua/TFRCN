function script_faster_rcnn_scut_vanilla()
% script_faster_rcnn_VOC2007_ZF()
% Faster rcnn training and testing with Zeiler & Fergus model
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 3;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

exp_name = 'faster_rcnn_scut_vanilla';
% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_Faster_RCNN;
% cache base
cache_base_proposal         = 'vgg16_vanilla';
cache_base_fast_rcnn        = 'vgg16_vanilla';
% train/test data
dataset                     = [];
use_flipped                 = false;
dataset                     = Dataset.scut_trainval(dataset, 'train02', use_flipped);
dataset                     = Dataset.scut_test(dataset, 'test25', use_flipped);

%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_t(model);
conf_fast_rcnn              = fast_rcnn_config_t(model);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);
% generate anchors and pre-calculate output size of rpn network
conf_proposal.exp_name = exp_name;
conf_fast_rcnn.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% test
dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_generate_proposal(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test        	= Faster_RCNN_Train.do_generate_proposal(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
% test
fprintf('\n***************\nstage one RPN test\n***************\n');
conf_proposal.method_name   = 'RPN-vanilla';
model.stage1_rpn.nms        = model.nms.test;
opts.miss                   = Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);

%% final test
fprintf('\n***************\nfinal test\n***************\n');
     
model.stage1_rpn.nms        = model.final_test.nms;
dataset.roidb_test       	= Faster_RCNN_Train.do_generate_proposal(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
conf_fast_rcnn.method_name  = 'Fast-vanilla';
model.stage1_fast_rcnn.nms  = model.nms.test;
opts.finalmiss              = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[3:5],...
                                    'exp_name', conf.exp_name);
end

function conf = proposal_config_t(model)
    conf = proposal_config('image_means', model.mean_image,...
                           'feat_stride', model.feat_stride);
    conf.eval_mul = false;
    %[10.^(-2:.25:0)] reference points (see bbGt>compRoc)
    conf.eval_ref = 10.^(-2:.25:0);
    
    conf.datasets = 'scut';
    % for eval_pLoad
    pLoad={'lbls',{'walk_person','ride_person'},'ilbls',{'people','person?',...
        'people?','squat_person'},'squarify',{3,.46}};
    pLoad = [pLoad 'hRng',[50 inf], 'vType',{{'none','partial'}},'xRng',[10 700],'yRng',[10 570]];
    conf.eval_pLoad = pLoad;
end

function conf = fast_rcnn_config_t(model)
    conf = fast_rcnn_config('image_means',   model.mean_image);
    conf.eval_mul = false;
    %[10.^(-2:.25:0)] reference points (see bbGt>compRoc)
    conf.eval_ref = 10.^(-2:.25:0);
    
    conf.datasets = 'scut';
    % for eval_pLoad
    pLoad={'lbls',{'walk_person','ride_person'},'ilbls',{'people','person?',...
        'people?','squat_person'},'squarify',{3,.46}};
    pLoad = [pLoad 'hRng',[50 inf], 'vType',{{'none','partial'}},'xRng',[10 700],'yRng',[10 570]];
    conf.eval_pLoad = pLoad;
end