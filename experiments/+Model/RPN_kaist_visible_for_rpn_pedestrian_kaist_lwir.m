function model = RPN_kaist_visible_for_rpn_pedestrian_kaist_lwir(exp_name, model)


model.mean_image                                = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'mean_image');
% model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
model.pre_trained_net_file                      = fullfile(pwd, 'models', exp_name, 'pre_trained_models', 'rpn_kaist_visible_vgg_16layers_stage1_rpn','train','final');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% stage 1 rpn, inited from pre-trained network
model.stage1_rpn.solver_def_file                = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_conv3_1', 'solver_60k80k.prototxt');
model.stage1_rpn.test_net_def_file              = fullfile(pwd, 'models', exp_name, 'rpn_prototxts', 'vgg_16layers_conv3_1', 'test.prototxt');
model.stage1_rpn.init_net_file                  = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN               = 10000;
model.stage1_rpn.nms.nms_overlap_thres       	= 0.5;
model.stage1_rpn.nms.after_nms_topN         	= 40;
end