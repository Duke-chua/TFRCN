function conf = fast_rcnn_config_pd(varargin)
% conf = fast_rcnn_config(varargin)
% Fast R-CNN configuration
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Faster R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2017, Zhewei Xu
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%
    ip = inputParser;
    
    %% training
    % whether use gpu
    ip.addParamValue('use_gpu',         gpuDeviceCount > 0, ...            
                                                        @islogical);  
    % Image scales -- the short edge of input image                                                
    ip.addParamValue('scales',          720,            @ismatrix);
    % Max pixel size of a scaled input image
    ip.addParamValue('max_size',        960,            @isscalar);
    % Images per batch
    ip.addParamValue('ims_per_batch',   2,              @isscalar);
    % Minibatch size
    ip.addParamValue('batch_size',      128,            @isscalar);
    % Fraction of minibatch that is foreground labeled (class > 0)
    ip.addParamValue('fg_fraction',     0.25,           @isscalar);
    % Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
    ip.addParamValue('fg_thresh',       0.5,            @isscalar);
    % Overlap threshold for a ROI to be considered background (class = 0 if
    % overlap in [bg_thresh_lo, bg_thresh_hi))
    ip.addParamValue('bg_thresh_hi',    0.5,            @isscalar);
    ip.addParamValue('bg_thresh_lo',    0.1,            @isscalar);
    % mean image, in RGB order
    ip.addParamValue('image_means',     128,            @ismatrix);
    % Use horizontally-flipped images during training?
    ip.addParamValue('use_flipped',     false,           @islogical);
    % Vaild training sample (IoU > bbox_thresh) for bounding box regresion
    ip.addParamValue('bbox_thresh',     0.5,            @isscalar);

    % random seed
    ip.addParamValue('rng_seed',        6,              @isscalar);


    %% testing
    ip.addParamValue('test_scales',     720,            @isscalar);
    ip.addParamValue('test_max_size',   960,            @isscalar);
    ip.addParamValue('test_nms',        0.3,            @isscalar);
    ip.addParamValue('test_binary',     false,          @islogical);
    
    %% evaluating
    ip.addParamValue('eval_mul',        false,          @islogical);
    %[10.^(-2:.25:0)] reference points (see bbGt>compRoc)
    ip.addParamValue('eval_ref',        10.^(-2:.25:0), @isvector);
    
    ip.parse(varargin{:});
    conf = ip.Results;

    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end