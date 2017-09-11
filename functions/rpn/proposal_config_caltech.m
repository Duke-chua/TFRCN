function conf = proposal_config_caltech(varargin)
% conf = proposal_config_caltech(varargin)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    ip = inputParser;
    
    %% training
    ip.addParamValue('use_gpu',         gpuDeviceCount > 0, ...            
                                                        @islogical);
                                    
%     % whether drop the anchors that has edges outside of the image boundary
%     ip.addParamValue('drop_boxes_runoff_image', ...
%                                         true,           @islogical);
                                    
   ip.addParamValue('drop_fg_boxes_runoff_image', ...
                                        true,           @islogical);
    
    % Image scales -- the short edge of input image                                                                                                
    ip.addParamValue('scales',          720,            @ismatrix);
    % Max pixel size of a scaled input image % for long edge
    ip.addParamValue('max_size',        960,           @isscalar);
    % Images per batch, only supports ims_per_batch = 1 currently
    ip.addParamValue('ims_per_batch',   1,              @isscalar);
    % Minibatch size
    ip.addParamValue('batch_size',      120,            @isscalar);
    % Fraction of minibatch that is foreground labeled (class > 0)
    ip.addParamValue('fg_fraction',     1/6,           @isscalar);
    % weight of background samples, when weight of foreground samples is
    % 1.0
    ip.addParamValue('bg_weight',       1.0,            @isscalar);
    % Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
    ip.addParamValue('fg_thresh',       0.5,            @isscalar);
    % Overlap threshold for a ROI to be considered background (class = 0 if
    % overlap in [bg_thresh_lo, bg_thresh_hi))
    ip.addParamValue('bg_thresh_hi',    0.5,            @isscalar);
    ip.addParamValue('bg_thresh_lo',    0,              @isscalar);
    % mean image, in RGB order
    ip.addParamValue('image_means',     256,            @ismatrix);
    % Use horizontally-flipped images during training?
    ip.addParamValue('use_flipped',     false,          @islogical);
    % Stride in input image pixels at ROI pooling level (network specific)
    % 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
    ip.addParamValue('feat_stride',     16,             @isscalar);
    % train proposal target only to labled ground-truths or also include
    % other proposal results (selective search, etc.)
    ip.addParamValue('target_only_gt',  true,           @islogical);

    % random seed                    
    ip.addParamValue('rng_seed',        6,              @isscalar);

    
    %% testing
    ip.addParamValue('test_scales',     720,            @isscalar);
    ip.addParamValue('test_max_size',   960,            @isscalar);
    ip.addParamValue('test_nms',        0.5,            @isscalar);
    ip.addParamValue('test_binary',     false,          @islogical);
    ip.addParamValue('test_min_box_size',16,            @isscalar);
    ip.addParamValue('test_min_box_height',50,          @isscalar);
    ip.addParamValue('test_drop_boxes_runoff_image', ...
                                        false,          @islogical);

    %% evaluating
    ip.addParamValue('eval_mul',        false,          @islogical);
    %[10.^(-2:.25:0)] reference points (see bbGt>compRoc)
    ip.addParamValue('eval_ref',        10.^(-2:.25:0), @isvector);
    
    ip.parse(varargin{:});
    conf = ip.Results;
    
    % for eval_pLoad
    pLoad = {'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};
    pLoad = [pLoad 'hRng',[50 inf],'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]];

    conf.eval_pLoad = pLoad;

    %assert(conf.ims_per_batch == 1, 'currently rpn only supports ims_per_batch == 1');
   
    assert(conf.scales == conf.test_scales);
    assert(conf.max_size == conf.test_max_size);
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end
