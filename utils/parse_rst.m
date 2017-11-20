function results = parse_rst(results, rst)
% results = parse_rst(results, rst)
% 解析结果
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
    % 如果results为空，则初始化
    if isempty(results)
        for i = 1:length(rst)
            % 按照rst(i).blob_name的名字对应的data初始化为[]
            results.(rst(i).blob_name).data = [];
        end
    end
    % 然后在rst(i).blob_name的名字的data增加一行rst的数据
    for i = 1:length(rst)
        results.(rst(i).blob_name).data = [results.(rst(i).blob_name).data; rst(i).data(:)];
    end
end