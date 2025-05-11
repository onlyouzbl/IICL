# IICL

# Unified Text Encoder with Intra- and Inter-modal Contrastive Learning for Cross-modal Recipe Retrieval

This is the PyTorch companion code for the paper:

*Bolin Zhang, Jianhao Li, Chao Yang, Bin Jiang, Takahiro Komamizu, Ichiro Ide, Jiangbo Qian.* **Unified Text Encoder with Intra- and Inter-modal Contrastive Learning for Cross-modal Recipe Retrieval**, TMM, 2025.

Note: The configuration file will be uploaded following the acceptance of the paper.


# Installation

- Create conda environment: 

```
conda env create -f environment.yml
```

- Activate it with: 

```
conda activate IICL
```



## Recipe1M Data preparation



- Download & uncompress Recipe1M [dataset](http://im2recipe.csail.mit.edu/dataset/download). The contents of the directory `DATASET_PATH` should be the following:

```
layer1.json
layer2.json
train/
val/
test/
```



The directories `train/`, `val/`, and `test/` must contain the image files for each split after uncompressing.

- Make splits and create vocabulary by running:

```
python preprocessing.py --root DATASET_PATH
```



This process will create auxiliary files under `DATASET_PATH/traindata`, which will be used for training.

 

## Training


- Launch training with:

```
CUDA_VISIBLE_DEVICES=0,1, python train.py --model_name model --root DATASET_PATH --save_dir saved_checkpoints --log_dir LOG_DIR 
```

Run `python train.py --help` for the full list of available arguments.

## Evaluation


- Extract features from the trained model for the test set samples of Recipe1M or Flickr30k:

```
python test.py --model_name model --eval_split test --root DATASET_PATH --save_dir saved_checkpoints
```

- Compute MedR and recall metrics for the extracted feature set:

```
python eval.py --embeddings_file saved_checkpoints/model/feats_test.pkl --retrieval_mode image2recipe --medr_N 1000
```



## Pretrained models



- We have provided the model trained on the Recipe1M dataset, please download and unzip it. Please download from the [link](https://pan.baidu.com/s/1o7fluqV2JUaz5lGWazJ7Nw?pwd=diz1).
- Extract features for the test set samples of Recipe1M using one of the pretrained models by running:

```
python test.py --model_name model --eval_split test --root DATASET_PATH --save_dir saved_checkpoints
```

- A file with extracted features will be saved under `../saved_checkpoints/MODEL_NAME`.

##  Acknowledgements

We are grateful for [image-to-recipe-transformers](https://github.com/amzn/image-to-recipe-transformers); it has been very helpful to us.
