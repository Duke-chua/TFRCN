# Thermal Faster R-CNN for FIR Pedestrians Detection on SCUT

### Introduction

This code is provided a modified Faster R-CNN for FIR pedestrians detection on SCUT Dataset.

The code RPN in this repo is written based on the MATLAB implementation of RPN+BF. Details about RPN+BF in: [zhangliliang/RPN_BF](https://github.com/zhangliliang/RPN_BF).

The code `external/code3.2.1` and `external/toolbox` is clone from [SCUT-CV/SCUT\_FIR\_Pedestrian\_Dataset](https://github.com/SCUT-CV/SCUT_FIR_Pedestrian_Dataset) which is based on  [Caltech dataset tool](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) and Piotr's Image & Video Matlab Toolbox. Detials about Piotr’s Toolbox are in：[pdollar/toolbox](https://github.com/pdollar/toolbox).

The SCUT FIR Pedestrians Dataset is a large far infrared pedestrian detection dataset. Detials about SCUT dataset are in: [SCUT-CV](http://www2.scut.edu.cn/cv/scut_fir_pedestrian_dataset/).

### Requirements

1. ubuntu (16.04 64bit)
2. MATLAB (our is MATLAB 2016b)
3. GPU: 1080ti or better

### Installation

1. Clone the TFRCN reposityory

   ```shell
   git clone --recursive https://github.com/xzhewei/TFRCN.git
   ```

2. Build Caffe

   In `./external/caffe ` directory, there is our used caffe version. Follow the [instruction](http://caffe.berkeleyvision.org/installation.html) to set up the prerequisites for Caffe. Use `make matcaffe` Build the mex file.

3. Download the [SCUT Dataset](https://github.com/SCUT-CV/SCUT_FIR_Pedestrian_Dataset)

   - Download the SCUT Dataset the videos into `./external/code3.2.1/data-scut/videos` directory
   - Download the SCUT Dataset the annotations into `./external/code3.2.1/data-scut/annotations` directory

4. Download the VGG-16 pretrain model in `VGG16_pretrain.zip` from [BaiduYun](https://pan.baidu.com/s/1mkcKVxu) or [GoogleDrive](https://drive.google.com/open?id=1noxehhM0SyzmBCIsMv--joIq37w_N1DK), and unzip it in the repo folder.

5. RUN `./startup()` and `./tfrcn_build()` 

   ​

### Training on SCUT

1. Start MATLAB from the repo folder

2. Training data preparation

   Extract image and annotation file into `./datasets`

   ```
   extract_img_anno_scut('./external/code3.2.1/data-scut','./datasets/scut/')
   ```

3. Run `script_tfrcn_train_scut` to train and test the TFRCN model on SCUT. The result will auto copy into `./external/code3.2.1/data-scut/res`

4. Run `dbEval_scut` , it would give the evaluation results on SCUT. The Reasonable MR is ~10%, Overall MR is ~33%.







 


