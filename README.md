# SSMTF

This repository contains the implementation of the following paper:
Multi-Scale Texture Fusion for Reference-based Image Super-Resolution: New Dataset and Solution

## Overview


## Dependencies and Installation

- Python >= 3.8
- PyTorch >= 2.2.1
- CUDA 12.1

## Dataset Preparation

- Train Set: [CUFED Dataset](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I)
- Test Set: [WR-SR Dataset](https://drive.google.com/drive/folders/16UKRu-7jgCYcndOlGYBmo5Pp0_Mq71hP?usp=sharing), [CUFED5 Dataset](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)

Please refer to [Datasets.md](datasets/DATASETS.md) for pre-processing and more details.

## Get Started

### Pretrained Models
Downloading the pretrained models from this [link](https://drive.google.com/drive/folders/1dTkXMzeBrHelVQUEx5zib5MdmvqDaSd9?usp=sharing) and put them under `experiments/pretrained_models folder`.

### Test

We provide quick test code with the pretrained model.

1. Modify the paths to dataset and pretrained model in the following yaml files for configuration.

    ```bash
    ./options/test/test_C2_matching.yml
    ./options/test/test_C2_matching_mse.yml
    ```

1. Run test code for models trained using **GAN loss**.

    ```bash
    python mmsr/test.py -opt "options/test/test_C2_matching.yml"
    ```

   Check out the results in `./results`.

1. Run test code for models trained using only **reconstruction loss**.

    ```bash
    python mmsr/test.py -opt "options/test/test_C2_matching_mse.yml"
    ```

   Check out the results in in `./results`


### Train

All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./experiments` and `./tb_logger` directory.

1. Modify the paths to dataset in the following yaml files for configuration.
   ```bash
   ./options/train/stage1_teacher_contras_network.yml
   ./options/train/stage2_student_contras_network.yml
   ./options/train/stage3_restoration_gan.yml
   ```

1. Stage 1: Train teacher contrastive network.
   ```bash
   python mmsr/train.py -opt "options/train/stage1_teacher_contras_network.yml"
   ```

1. Stage 2: Train student contrastive network.
   ```bash
   # add the path to *pretrain_model_teacher* in the following yaml
   # the path to *pretrain_model_teacher* is the model obtained in stage1
   ./options/train/stage2_student_contras_network.yml
   python mmsr/train.py -opt "options/train/stage2_student_contras_network.yml"
   ```

1. Stage 3: Train restoration network.
   ```bash
   # add the path to *pretrain_model_feature_extractor* in the following yaml
   # the path to *pretrain_model_feature_extractor* is the model obtained in stage2
   ./options/train/stage3_restoration_gan.yml
   python mmsr/train.py -opt "options/train/stage3_restoration_gan.yml"

   # if you wish to train the restoration network with only mse loss
   # prepare the dataset path and pretrained model path in the following yaml
   ./options/train/stage3_restoration_mse.yml
   python mmsr/train.py -opt "options/train/stage3_restoration_mse.yml"
   ```



