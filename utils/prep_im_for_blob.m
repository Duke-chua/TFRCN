function [im, im_scale] = prep_im_for_blob(im, im_means, target_size, max_size)
    im = single(im); % im 转换为 single 单精度
    
    if ~isa(im, 'gpuArray')
        try
            im = bsxfun(@minus, im, im_means); % im 和 im_means对应元素相减
        catch % im 和 im_means 的尺寸不匹配
            % im_means 按照 im 的尺寸进行resize，进行双线性差值、抗锯齿
            im_means = imresize(im_means, [size(im, 1), size(im, 2)], 'bilinear', 'antialiasing', false);
            im = bsxfun(@minus, im, im_means);
        end
        im_scale = prep_im_for_blob_size(size(im), target_size, max_size);

        target_size = round([size(im, 1), size(im, 2)] * im_scale);
        im = imresize(im, target_size, 'bilinear', 'antialiasing', false);
    else
        % for im as gpuArray 可能是imresize函数的差异
        try
            im = bsxfun(@minus, im, im_means);
        catch
            % im_means的最大缩放比例
            im_means_scale = max(double(size(im, 1)) / size(im_means, 1), double(size(im, 2)) / size(im_means, 2));
            im_means = imresize(im_means, im_means_scale);    
            y_start = floor((size(im_means, 1) - size(im, 1)) / 2) + 1;
            x_start = floor((size(im_means, 2) - size(im, 2)) / 2) + 1;
            im_means = im_means(y_start:(y_start+size(im, 1)-1), x_start:(x_start+size(im, 2)-1));
            im = bsxfun(@minus, im, im_means);
        end
        
        im_scale = prep_im_for_blob_size(size(im), target_size, max_size);
        im = imresize(im, im_scale);
    end
end