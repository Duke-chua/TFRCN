function blob = im_list_to_blob(ims)
% 将图像矩阵转换为尺寸相同的blob
% INPUT
%   - ims {} 每个元素是一个图像三维图像矩阵
% OUTPUT
%   - blob [] MxNx3xNUM
% -------------------------------------------------------
% Copyright (c) 2017, Zhewei Xu
% -------------------------------------------------------
    % 对每个img求其尺寸大小，并获得最大的宽和高
    max_shape = max(cell2mat(cellfun(@size, ims(:), 'UniformOutput', false)), [], 1);
    % 确保图像的第三维是3，即RGB
    assert(all(cellfun(@(x) size(x, 3), ims, 'UniformOutput', true) == 3));
    num_images = length(ims);
    % 初始化blob，按最大的宽、最大的高建立
    blob = zeros(max_shape(1), max_shape(2), 3, num_images, 'single');
    %赋值
    for i = 1:length(ims)
        im = ims{i};
        blob(1:size(im, 1), 1:size(im, 2), :, i) = im; 
    end
end