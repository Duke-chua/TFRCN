function script_rpn_pedestrian_VGG16_scut()
% script_rpn_pedestrian_VGG16_scut()
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
opts.gpu_id                 = 3;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

exp_name = 'VGG16_scut';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_rpn_pedestrian('VGG16_caltech');
% cache base
cache_base_proposal         = 'vgg16_skip02_overall';
% train/test data
dataset                     = [];
use_flipped                 = false;
dataset                     = Dataset.scut_trainval(dataset, 'train02', use_flipped);
dataset                     = Dataset.scut_test(dataset, 'test25', use_flipped);

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config(model);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder_rpn(cache_base_proposal, model);
% generate anchors and pre-calculate output size of rpn network 
conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);
                
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_pd(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

%% test
conf_proposal.method_name = 'RPN-skip02-overall';
Faster_RCNN_Train.do_proposal_test_pd(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_pd(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_pd(cache_name, ...
                                    'scales',  2.6*(1.3.^(0:8)),...%[29 34 39 44 51 59 71 89 126]./16.*sqrt(0.46),...% anchors scales [2.6000    3.3800    4.3940    5.7122    7.4259    9.6536   12.5497   16.3146   21.2090]
                                    'ratios',  [1 / 0.46], ... % pedestrain high with ratio
                                    'exp_name', conf.exp_name);
end

function conf = proposal_config(model)
    conf = proposal_config_pd('image_means', model.mean_image,...
                              'feat_stride', model.feat_stride ...
                             ,'scales',        576   ...
                             ,'max_size',      720   ...
                             ,'ims_per_batch', 1     ...
                             ,'batch_size',    120   ...
                             ,'fg_fraction',   1/6   ...
                             ,'bg_weight',     1.0   ...
                             ,'fg_thresh',     0.5   ...
                             ,'bg_thresh_hi',  0.5   ...
                             ,'bg_thresh_lo',  0     ...
                             ,'test_scales',   576   ...
                             ,'test_max_size', 720   ...
                             ,'test_nms',      0.5   ...
                             ,'test_min_box_size',0 ...
                             ,'test_min_box_height',20 ...
                             ,'datasets',     'scut' ...
                              );
    % for eval_pLoad
    pLoad={'lbls',{'walk_person','ride_person'},'ilbls',{'people','person?',...
       'people?','squat_person'},'squarify',{3,.46}};
    pLoad = [pLoad 'hRng',[50 inf], 'vType',{{'none','partial'}},'xRng',[10 700],'yRng',[10 570]];
    conf.eval_pLoad = pLoad;
end