function model = set_cache_folder_rpn(cache_base_proposal, cache_base_fast_rcnn, model)
% model = set_cache_folder_rpn(cache_base_proposal, model)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    model.rpn.cache_name = [cache_base_proposal, '_stage1_rpn'];

    model.fast_rcnn.cache_name = ...
            [cache_base_proposal, ...
            strrep(sprintf('_top%d_nms%g_top%d', model.rpn.nms.per_nms_topN, ...
            model.rpn.nms.nms_overlap_thres, model.rpn.nms.after_nms_topN), '.', '_'), ...
            cache_base_fast_rcnn, '_stage1_fast_rcnn'];
end