function im_scale = prep_im_for_blob_size(im_size, target_size, max_size)
% 准备缩放im
% INPUT
%     - im_size [2] 图像的尺寸，例如[480 640]
%     - target_size [1] 短边缩放后的目标尺寸，例如 720
%     - max_size [1] 长边的最大尺寸 例如 960
% OUPUT 
%     - im_scale [1] 缩放因子，double类型
% EXAMPLE
%     im_scale = prep_im_for_blob_size([480 640], 720, 960)
% -------------------------------------------------------
% Copyright (c) 2017, Zhewei Xu
% -------------------------------------------------------
    im_size_min = min(im_size(1:2));
    im_size_max = max(im_size(1:2));
    % 根据图像短边进行缩放
    im_scale = double(target_size) / im_size_min;
    
    % Prevent the biggest axis from being more than MAX_SIZE
    % 防止缩放后长边超过最大尺寸
    if round(im_scale * im_size_max) > max_size
        im_scale = double(max_size) / double(im_size_max);
    end
end