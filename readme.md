



## Image-Text Aggregation for Open-Vocabulary Semantic Segmentation

![me_1](C:\Users\22980\Desktop\ita_code\me_1.png)

This paper proposes a novel single-stage open-vocabulary semantic segmentation method based on image-text aggregation (ITA). In contrast to two-stage approaches, we unify the mask generation and classification operations into a single stage by designing an image-text aggregation module, which improves the ability to handle open-vocabulary segmentation tasks. Additionally, to further employ image-text aggregation, we adopt a detail enhancement module to alleviate the problem of detail loss caused by downsampling in high-resolution images. Furthermore, we leverage a dominant category unearthing module to mitigate the issue of random optimization resulting from the random initialization of query vectors. Experimental results on five widely used benchmark datasets demonstrate that our ITA achieves excellent segmentation performance compared to state-of-the-art open-vocabulary semantic segmentation methods.



### Installation

```
1.
conda create --name ita python=3.8 -y
conda activate ita
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
```

```
2.
git clone https://github.com/facebookresearch/detectron2.git
if error you can:
wget https://github.com/facebookresearch/detectron2/archive/refs/heads/main.zip
and unzip this zip file
cd detectron2-main
pip install -e .
cd ..
```

```
3.
pip install git+https://github.com/cocodataset/panopticapi.git
if error you can:
wget https://github.com/cocodataset/panopticapi/archive/refs/heads/master.zip
and unzip this zip file
cd panopticapi-master/
python setup.py install
cd ..
```

```
4.
git clone https://github.com/bytedance/fcclip.git
if error you can:
wget https://github.com/mcordts/cityscapesScripts/archive/refs/heads/master.zip
and unzip this zip file 
cd cd cityscapesScripts-master/
python setup.py install
cd ..
```

```
cd ita
pip install -r requirements.txt
```

### Dateset 

see preparing datasets for ITA.

### Started

#### Training & Evaluation in Command Line

To train a model with "train_net.py", first read the readme.md, then run:

python train_net.py --num-gpus 8 --config-file configs/coco/semantic-segmentation/ita/fcclip_convnext_large_eval_ade20k.yaml

#### To evaluate a model's performance, use

python train_net.py \
  --config-file configs/coco/semantic-segmentation/ita/ita_convnext_large_eval_ade20k.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file