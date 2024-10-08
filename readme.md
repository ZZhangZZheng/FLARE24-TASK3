#  Solution of Team liuhuahua for FLARE24TASK3 Challenge

>Unsupervised Domain Adaptive Segmentation with Single-content Multi-style Generation and Simplified Pseudo-label Selection
_Xiao Luan and Zheng Zhang and Weiqiang Wang and Xiongfeng Huang and Yue Zeng_

Built upon [Kaiseem/DAR-UNet](https://github.com/Kaiseem/DAR-UNet) and [Ziyan-Huang/FLARE22](https://github.com/Ziyan-Huang/FLARE22), this repository provides the solution of team blackbean for [MICCAI FLARE24TASK3](https://www.codabench.org/competitions/2296/) Challenge. The details of our method are described in our [paper](https://openreview.net/forum?id=705SUzdm3p).


## Methods

Our framework is shown below：
![framework]()
We use the [Evaluation](https://github.com/JunMa11/FLARE/tree/main/FLARE22/Evaluation) to eval our work.
## Datasets

We train the model on the dataset provided by FLARE24.
The training dataset is curated from more than 30 medical centers under the license permission, including TCIA , LiTS , MSD , KiTS , autoPET , AMOS , LLD-MMRI , TotalSegmentator , and AbdomenCT-1K , and past FLARE Challenges . The training set includes 2050 abdomen CT scans and over 4000 MRI scans. The validation and testing sets include 110 and 300 MRI scans, respectively, which cover various MRI sequences, such as T1, T2, DWI, and so on.
We only use the MRI scans and 50 CT scans annotated by people amog the 2050 abdomen CT scans.



## Results

Our method achieved an average score of 63.41% and 68.08%([FLARE24TASK3](https://www.codabench.org/competitions/2296/#/results-tab)) for the lesion DSC and NSD on the validation dataset, respectively. The average running time and area under GPU memory-time curve are 10.36s and 13331MB, respectively.
Here are our precision metrics and resource consumption on the validation set：
![the metrics of the ablation study]()
![the resource consumption]()
# Code for the challenge

You can reproduce our method as follows step by step:

## Environments and Requirements:
-   You can prepare two environments for the **Generation** and the **Segmentation** respectively. Or you can follow:
- Prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the generation dependencies.
```
pip install -r requirements.txt
```
- In the previously established environment, you should meet the requirements of nnUNet. For more details, please refer to [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
There will be no conflict between them.


## 1 Generation

Our two-stage framework was inspired by [DAR-UNet](https://github.com/Kaiseem/DAR-UNet).
```
cd Single-content Multi-style Generation
```
### 1.1  prepare_data
```
cd datasets
python prepare_data_for_FLARE.py
```

### 1.2  Train stage one Single-content Multi-style Generation translation model for style transferynchronization

```
python stage_1_i2i_train.py --name sourceAtotargetB
```


### 1.3 Generate target-like source domain images

```
python stage_1.5_i2i_inference.py --ckpt_path YOUR_G_CKPT_PATH --source_npy_dirpath SOURCE_PATH --target_npy_dirpath TARGET_PATH --save_npy_dirpath SAVE_PATH --k_means_clusters 8
```
>YOUR_G_CKPT_PATH: the XXX.pt from 1.2
SOURCE_PATH: the source_training_npy from 1.1
TARGET_PATH: like:
 Flare/FLARE24-Task3-MR/Training/LLD-MMRI-3984
SAVE_PATH: like: nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22
```
```

## 2 Segmentation
Our segmentation  follow  [Ziyan-Huang/FLARE22](https://github.com/Ziyan-Huang/FLARE22).
```
cd Segmentation
```


### 2.1 Training Big nnUNet for Pseudo Labeling

- 2.1.1 Copy the following files in this repo to your nnUNet environment.
```
FLARE24TASK3/Segmentation/nnunet/training/network_training/nnUNetTrainerV2_FLARE.py
FLARE24TASK3/Segmentation/nnunet/experiment_planning/experiment_planner_FLARE22Big.py
```
- 2.1.2 Prepare 50*8 Generated labeled Data of FLARE24
```
nnUNet_raw_data_base/nnUNet_raw_data/Task022_FLARE22/
├── dataset.json
├── imagesTr
├── imagesTs
└── labelsTr
```
> **Note:** You can use getjson.py to get dataset.json
- 2.1.3 Conduct automatic preprocessing using nnUNet and  train Big nnUNet by 5-fold Cross Validation
```
nnUNet_plan_and_preprocess -t 22 -pl3d ExperimentPlanner3D_FLARE22Big -pl2d None
```
```
for FOLD in 0 1 2 3 4
do
nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Big 22 $FOLD -p nnUNetPlansFLARE22Big
done
```
-2.1.4  Generate Pseudo Labels for 3984+883 Unlabeled Data
```
nnUNet_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER  -t 22  -tr nnUNetTrainerV2_FLARE_Big  -m 3d_fullres  -p nnUNetPlansFLARE22Big  --all_in_gpu True 
```

### 2.2  Filter Low-quality Pseudo Labels
We removed the 1348 low-quality scans through the simplified pseudo-label selection method
```
python amos_lld.py
```
### 2.3  Train Small nnUNet
- 2.3.1 Copy the following files in this repo to your nnUNet environment.
 ```
FLARE24TASK3/Segmentation/nnunet/training/network_training/nnUNetTrainerV2_FLARE.py
FLARE24TASK3/Segmentation/nnunet/experiment_planning/experiment_planner_FLARE22Small.py
```
- 2.3.2 Prepare 50*8 Labeled Data and 3984+883-1348 Selected Pseudo Labeled Data of FLARE24
- 2.3.3 Conduct automatic preprocessing using nnUNet
```
nnUNet_plan_and_preprocess -t 26 -pl3d ExperimentPlanner3D_FLARE22Small -pl2d None
```
- 2.3.4  Train small nnUNet on all training data
```
nnUNet_train 3d_fullres nnUNetTrainerV2_FLARE_Small 26 all -p nnUNetPlansFLARE22Small
```
### 2.4  Do Efficient Inference with Small nnUNet
```
nnUNet_predict -i INPUT_FOLDER  -o OUTPUT_FOLDER  -t 26  -p nnUNetPlansFLARE22Small   -m 3d_fullres \
 -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta
```

# Acknowledgement
We thank the contributors of datasets and the challenge organizers.
# Docker Link
You can get the [docker](https://pan.baidu.com/s/11RM32jPOBzFmzeGBH6sB7A?pwd=8888) .If you have any questions, please send us email 
```
s230201153@stu.cqupt.edu.cn
``` 
