# 基于RPN的热红外行人检测方法

By Zhewei Xu

### Introduction

本项目在RPN+BF代码的基础上进行修改，应用于KAIST热红外视频图像。

### 实验

#### 1. 用Caltech预训练的RPN，在KAIST-visible上测试
- 召回率
- 漏检率
- ROC

#### 2. 在KAIST-visible上训练RPN-kv
- 召回率
- 漏检率
- ROC

#### 3. 用RPN-kv在KAIST-lwir上测试
- 召回率
- 漏检率
- ROC

#### 4. 在KAIST-lwir上训练RPN-kl
- 召回率
- 漏检率
- ROC

#### 5. 在KAIST-lwir上训练，用RPN-kv作为预训练模型，得到RPN-kv-kl
- 召回率
- 漏检率
- ROC

#### 6. 在KAIST-lwir-flipped上训练，用RPN-kv作为预训练模型，得到RPN-kv-kl-flip
- 召回率
- 漏检率
- ROC
 


