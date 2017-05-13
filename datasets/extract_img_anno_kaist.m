% extract_img_anno()
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
pth = '/mnt/RD/DataSet/KAIST/data-kaist-lwir/';
tDir = './datasets/kaist_lwir/';

for s=1:2
  if(s==1), type='test'; skip=[]; else type='train'; skip=3; end
  dbInfo3(['kaist-lwir-all-' type]);
  if(exist([tDir type '/annotations'],'dir')), continue; end
  dbExtract3(pth,[tDir type],'lwir',1,skip);
end

