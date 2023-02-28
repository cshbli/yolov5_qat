# Yolov5 Quantization Aware Training QAT

## Notes

**This repo is based on the release v7.0 of [yolov5](https://github.com/ultralytics/yolov5/).**

## 0. Prepare Environment

- This version QAT requires PyTorch version 1.9.1.
  - For YOLOv5, before install requirements.txt, please install torchvision version 0.10.1 first
    ```
    pip install torchvision==0.10.1
    pip install -r requirements.txt  
    ```

### 0.1 Create a virtual environment with PyTorch 1.9.1

```
python3 -m venv ./venv/torch1.9.1

source ~/venv/torch1.9.1/bin/activate

pip install setuptools numpy==1.23.5

pip install torch==1.9.1

pip install torchvision==0.10.1
```

### 0.2 QAT setup

- Remove the Tensorflow installation

```
install_requires.extend([
    'torch==1.9.1',
    'torchviz',
    'onnx==1.9.0',
    'onnxoptimizer==0.2.6',
    'onnxruntime-gpu==1.9.0',
    'tf2onnx==1.9.2',
    # 'tensorflow==1.15.2',
    'tqdm'
])
```

- Setup
```
pip install Cython

python setup.py develop
```

- In case there are some errors:
```
AttributeError: module 'numpy' has no attribute 'object'
```

It seems numpy 1.24.0 has some errors. Try:
```
pip install numpy==1.23.5

1 - pip uninstall -y numpy
2 - pip uninstall -y setuptools
3 - pip install setuptools
4 - pip install numpy
```

### 0.3 YOLOv5 installation
```
pip install -r requirements.txt
```

## 1 Setup

### 1.1 Clone the Sample  
```
git clone https://github.com/cshbli/yolov5_qat.git
```  

### 1.2 Dataset Preparation

Download the labels and images of coco2017, and unzip to the same level directory as the current project. 

```
Projects
├──datasets
|   └── coco                 # Directory for datasets 
│       ├── annotations
│       │   └── instances_val2017.json
│       ├── images
│       │   ├── train2017
│       │   └── val2017
│       ├── labels
│       │   ├── train2017
│       │   └── val2017
│       ├── train2017.txt
│       └── val2017.txt
└── yolov5_qat               # Quantization source code 
```

```
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip         # Download the labels needed
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```  

### 1.3 Download Yolov5m Pretrained Model  

```bash
$ cd /Projects/yolov5_qat
$ cd weights
$ wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
$ cd ..
```  

#### Check this pretrained model accuracy

```
python val.py --weights weights/yolov5m.pt --data coco.yaml
```

Outputs: 
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.644
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.489
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.278
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.504
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.632
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777
```

### 1.4 Replacing SiLU with ReLU

- Make sure to change the learning rate, otherwise it will long time to converge.
  - We use a new hyps yaml here [hyp.m-relu-tune.yaml](./hyp.m-relu-tune.yaml). It is based on `hyp.scratch-low.yaml`, changed lr to smaller value.
    ```
    lr0: 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3), changed from 0.01
    lrf: 0.001  # final OneCycleLR learning rate (lr0 * lrf), changed from 0.01
    ...
    warmup_bias_lr: 0.01  # warmup initial bias lr, changed from 0.1
    ...
    ```
- Disable GIT info checking
- Once we changed the default_act to ReLU, we can't use auto batch size anymore. 
    - We need specifiy the `batch-size`
    - Also we can change the default `batch-size` from 16 to 64

It takes a long time to complete the retraining, please be patient.

```
python train.py --data coco.yaml --epochs 50 --weights weights/yolov5m.pt --hyp data/hyps/hyp.m-relu-tune.yaml --batch-size 64
```

```
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/49      6.16G    0.04115    0.06202    0.01698        150        640: 100%|██████████| 1849/1849 [51:50<00:00,  1.68s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:26<00:00,  2.17s/it]
                   all       5000      36335      0.701      0.557      0.609      0.416
    
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/49      10.2G    0.03954    0.05978    0.01563        198        640: 100%|██████████| 1849/1849 [51:32<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.709      0.567      0.617      0.428

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      14/49      10.2G    0.03948    0.05968    0.01557        240        640: 100%|██████████| 1849/1849 [51:30<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.708      0.568      0.618      0.429

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/49      10.2G    0.03922    0.05922    0.01519        162        640: 100%|██████████| 1849/1849 [51:23<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.713      0.567       0.62       0.43

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      20/49      10.2G    0.03911    0.05934    0.01513        228        640: 100%|██████████| 1849/1849 [51:33<00:00,  1.67s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 40/40 [01:25<00:00,  2.13s/it]
                   all       5000      36335      0.707      0.569      0.619      0.431
```

Here is the complete retraining log file [retraining after replacing SiLU with ReLU](./notes/relu_retraining.csv).

Assuming the retraining result folder name is changed to **relu**, run validation test:

```
python val.py --weights runs/train/relu/weights/best.pt --data coco.yaml

```

We will get the following validation results: 

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.625
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.613
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.767
Results saved to runs/val/exp
```

### 1.5 Replacing SiLU with ReLU6 (Optional)

Similar as replacing SiLU with ReLU

```
python train.py --data coco.yaml --epochs 50 --weights weights/yolov5m.pt --hyp data/hyps/hyp.m-relu-tune.yaml --batch-size 64
```

Assuming the retraining result folder name is changed to **relu6**, run validation test:

```
python val.py --weights runs/train/relu6/weights/best.pt --data coco.yaml

```

We will get the following validation results: 

```
Fusing layers... 
Model summary: 212 layers, 21172173 parameters, 0 gradients, 48.9 GFLOPs
val: Scanning /home/hongbing/Projects/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:15<00:00,  2.07it/s]
                   all       5000      36335      0.701      0.563      0.615      0.428
Speed: 0.1ms pre-process, 10.1ms inference, 0.9ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp3/best_predictions.json...
loading annotations into memory...
Done (t=0.29s)
creating index...
index created!
Loading and preparing results...
DONE (t=3.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=46.18s).
Accumulating evaluation results...
DONE (t=10.45s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.621
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.260
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.763
```

### 1.6 Code changes

Please refer to all related code changes here [code change log](./notes/bst_code_changes.md).

## 2. QAT

BST QAT flow chart:

<img src="notes/bst_qat_flow.png">

### Model fusing debug and pre-bind debug

We can turn on the `debug_mode=True` to print out the model structure while doing `fuse_modules`.

```
# use CPU on input_tensor as our backend for parsing GraphTopology forced model to be on CPU    
model = quantizer.fuse_modules(model, auto_detect=True, debug_mode=False, input_tensor=sample_data.to('cpu'))
```

The example [fuse modules pdf printout](./notes/fuse_modules_debug.gv.pdf).

We can turn on the `debug_mode=True` to print out the model structure while doing `pre_bind`.

```
# 1) [bst_alignment] get b0 pre-bind qconfig adjusting Conv's activation quant scheme
pre_bind_qconfig = quantizer.pre_bind(model, input_tensor=sample_data.to('cpu'), debug_mode=False,
    observer_scheme_dict={"weight_scheme": "MovingAveragePerChannelMinMaxObserver", 
                          "activation_scheme": "MovingAverageMinMaxObserver"})
```

The example [pre-bind pdf printout](./notes/pre_bind_debug.gv.pdf).


### Experiment 1: Quantization with Conv+BN+ReLU, skip_add and Concat

- QAT can't use multiple GPUs. We need to specify the device ID.
- Please see the quantized model structure here: [Quantized mode structure](./notes/bst_qat_model.txt)

- Stop observers after epoch 0.
- Power of 2 scale with rounding. 

```
python train.py --data coco.yaml --epochs 20 --cfg models/yolov5m.yaml \
--weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.m-relu-tune.yaml \
--batch-size 32 --qat --device 1
```

Result log: 

```
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       0/19        15G     0.0431    0.05806     0.0141        199        640: 100%|██████████| 3697/3697 [1:08:30<00:00,  1.11s/it]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [02:11<00:00,  1.67s/it]
                   all       5000      36335      0.677      0.538      0.584      0.369

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/19      16.8G     0.0429    0.05798    0.01391        169        640: 100%|██████████| 3697/3697 [58:17<00:00,  1.06it/s] 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [02:11<00:00,  1.66s/it]
                   all       5000      36335      0.684      0.552      0.595       0.38
      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size      
      18/19      18.2G    0.04176    0.05679    0.01283        216        640: 100%|██████████| 3697/3697 [36:50<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.32it/s]
                   all       5000      36335      0.686      0.555        0.6      0.387

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size      
      19/19      18.2G    0.04173    0.05679    0.01279        198        640: 100%|██████████| 3697/3697 [36:33<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.686      0.556      0.602       0.39
```

### Experiment 2: Quantization with full pipeline, Conv+BN+ReLU, skip_add and Concat, and alignment

- QAT can't use multiple GPUs. We need to specify the device ID.
- Please see the quantized model structure here: [Quantized mode structure](./notes/bst_qat_model.txt)

- Stop observers after epoch 0, free batch norm after epoch 0
- Power of 2 scale with rounding. 

```
python train.py --data coco.yaml --epochs 20 --cfg models/yolov5m.yaml \
--weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.m-relu-tune.yaml \
--batch-size 32 --qat --device 1
```

Result log: 

```
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       0/19      14.6G    0.04198    0.05757     0.0138        199        640: 100%|██████████| 3697/3697 [37:51<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.26it/s
                   all       5000      36335      0.691      0.551      0.598      0.387
                   
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       1/19      23.1G     0.0422    0.05728    0.01337        169        640: 100%|██████████| 3697/3697 [36:44<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.693      0.546      0.599      0.381

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       2/19      23.1G    0.04257    0.05747    0.01341        144        640: 100%|██████████| 3697/3697 [36:40<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:01<00:00,  1.29
                   all       5000      36335       0.68      0.545       0.59      0.372

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       3/19      23.1G     0.0426    0.05772    0.01357        149        640: 100%|██████████| 3697/3697 [36:38<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335      0.684      0.547      0.595      0.375

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       4/19      23.1G    0.04255    0.05764    0.01357        197        640: 100%|██████████| 3697/3697 [36:40<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:01<00:00,  1.29
                   all       5000      36335      0.693      0.543      0.595       0.38
      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      10/19      23.1G    0.04202    0.05711    0.01304        180        640: 100%|██████████| 3697/3697 [36:35<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:01<00:00,  1.29
                   all       5000      36335      0.687      0.556      0.602      0.387

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/19      23.1G    0.04181    0.05672    0.01283        198        640: 100%|██████████| 3697/3697 [36:30<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335      0.686      0.553        0.6      0.389
                   
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      15/19      23.1G    0.04173    0.05668    0.01279        183        640: 100%|██████████| 3697/3697 [36:37<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335      0.702      0.552      0.604      0.391
      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      16/19      23.1G    0.04159    0.05661    0.01271        228        640: 100%|██████████| 3697/3697 [36:35<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335        0.7      0.548      0.601      0.386

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      17/19      23.1G    0.04152    0.05646    0.01263        176        640: 100%|██████████| 3697/3697 [36:30<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335      0.696      0.553      0.602      0.393
      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      18/19      23.1G    0.04144    0.05639    0.01257        216        640: 100%|██████████| 3697/3697 [36:38<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335      0.684       0.56      0.602       0.39

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/19      23.1G    0.04138    0.05634     0.0125        198        640: 100%|██████████| 3697/3697 [36:37<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30
                   all       5000      36335      0.701      0.555      0.606      0.394
```

### Experiment 3: Quantization with full pipeline, Conv+BN+ReLU, skip_add and Concat, and alignment

- Same as Experiment 2, but with more epochs
- Model collapsed

```
python train.py --data coco.yaml --epochs 100 --cfg models/yolov5m.yaml \
--weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.m-relu-tune.yaml \
--batch-size 32 --qat --device 2
```

Result log: 

```
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       0/19      14.6G    0.04198    0.05757     0.0138        199        640: 100%|██████████| 3697/3697 [37:51<00:00,  1.63it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.26it/s
                   all       5000      36335      0.691      0.551      0.598      0.387
                   
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      32/99      23.3G    0.04298    0.05868    0.01414        177        640: 100%|██████████| 3697/3697 [37:13<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.26it/s]
                   all       5000      36335      0.687      0.535      0.586      0.376

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size      
      33/99      23.3G    0.05581    0.06695    0.03414        180        640: 100%|██████████| 3697/3697 [37:15<00:00,  1.65it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [02:11<00:00,  1.67s/it]
                   all       5000      36335   3.84e-05     0.0026   2.18e-05   6.22e-06

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      34/99      23.3G     0.1017    0.09084     0.1002        175        640: 100%|██████████| 3697/3697 [36:59<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:26<00:00,  1.10s/it]
                   all       5000      36335    4.7e-05    0.00197   2.77e-05   7.73e-06

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      35/99      23.3G    0.09932    0.08901    0.09485        173        640: 100%|██████████| 3697/3697 [36:58<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:04<00:00,  1.22it/s]
                   all       5000      36335   2.38e-05    0.00206   1.34e-05   4.05e-06

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      36/99      23.3G    0.09783    0.08883    0.09386        198        640: 100%|██████████| 3697/3697 [37:01<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335    2.4e-05    0.00171   1.43e-05   4.32e-06
```

### Experiment 4: Quantization with full pipeline, Conv+BN+ReLU, skip_add and Concat, and alignment, and ReLU6

- Batch Norm already folded before QAT starts.
- Disable observers after 1st epoch.
- Has exploding gradient issues.

```
python train.py --data coco.yaml --epochs 100 --cfg models/yolov5m.yaml \
--weights runs/train/relu6/weights/best.pt --hyp data/hyps/hyp.m-relu-tune.yaml \
--batch-size 32 --qat --device 0
```

Result log: 

```
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       0/99      14.4G    0.04153    0.05709    0.01401        199        640: 100%|██████████| 3697/3697 [38:13<00:00,  1.61it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:04<00:00,  1.22it/s]
                   all       5000      36335      0.696      0.555      0.601      0.391

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/99      16.9G    0.04148    0.05676    0.01385        366        640:  33%|███▎      | 1220/3697 [12:16<24:55,  1.66it/s]train.py:409: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       1/99      16.9G     0.0418    0.05699    0.01384        169        640: 100%|██████████| 3697/3697 [37:06<00:00,  1.66it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.26it/s]
                   all       5000      36335      0.692      0.553      0.597      0.377
      
      10/99      18.3G    0.04204    0.05741    0.01371        180        640: 100%|██████████| 3697/3697 [36:56<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.26it/s]
                   all       5000      36335      0.687      0.549      0.595      0.385      
      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      20/99      18.3G     0.0422     0.0577    0.01373        384        640:  77%|███████▋  | 2837/3697 [28:23<08:37,  1.66it/s]train.py:409: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
      20/99      18.3G     0.0422    0.05759    0.01372        205        640: 100%|██████████| 3697/3697 [36:59<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.26it/s]
                   all       5000      36335        0.7      0.544      0.596      0.384

      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      34/99      18.3G    0.04265    0.05866    0.01427        432        640:  86%|████████▋ | 3197/3697 [31:55<04:57,  1.68it/s]train.py:409: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
      34/99      18.3G    0.04267    0.05865    0.01426        175        640: 100%|██████████| 3697/3697 [36:54<00:00,  1.67it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:01<00:00,  1.28it/s]
                   all       5000      36335      0.689      0.539      0.593      0.381
```

### Experiment 5: Quantization with Conv+BN+ReLU, allow observers and Batch Norm all the time

- Has exploding gradient issues.

```
python train.py --data coco.yaml --epochs 100 --cfg models/yolov5m.yaml --weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.m-relu-tune.yaml --batch-size 16 --qat --device 1
```

Result log: 

```
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/99      11.8G    0.04318    0.05782    0.01436        199        640: 100%|██████████| 7393/7393 [1:04:26<00:00,  1.91it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:20<00:00,  1.96it/s]
                   all       5000      36335      0.665      0.524      0.568      0.352

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/99        11G    0.04357    0.05801     0.0144        169        640: 100%|██████████| 7393/7393 [1:02:33<00:00,  1.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:18<00:00,  1.99it/s]
                   all       5000      36335      0.681      0.537      0.581      0.359

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       2/99        11G    0.04404    0.05837    0.01475         88        640:  34%|███▍      | 2515/7393 [21:10<40:23,  2.01it/s]train.py:411: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       2/99        11G    0.04438    0.05886    0.01498        144        640: 100%|██████████| 7393/7393 [1:02:05<00:00,  1.98it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:17<00:00,  2.02it/s]
                   all       5000      36335      0.667       0.52      0.562      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       3/99        11G    0.04522    0.06006    0.01587        249        640:  86%|████████▋ | 6377/7393 [53:44<08:34,  1.97it/s]train.py:411: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       3/99        11G    0.04524    0.06006     0.0159        149        640: 100%|██████████| 7393/7393 [1:02:25<00:00,  1.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:19<00:00,  1.98it/s]
                   all       5000      36335      0.664      0.518      0.564      0.345

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       4/99        11G     0.0457    0.06033    0.01618        197        640: 100%|██████████| 7393/7393 [1:02:33<00:00,  1.97it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:25<00:00,  1.83it/s]
                   all       5000      36335       0.66      0.523      0.564      0.346
      
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       9/99      11.1G     0.0463    0.06044    0.01618        240        640: 100%|██████████| 7393/7393 [1:05:33<00:00,  1.88it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:22<00:00,  1.90it/s]
                   all       5000      36335      0.629       0.52      0.552      0.336

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      11/99      11.1G    0.04646    0.06059     0.0162        162        640: 100%|██████████| 7393/7393 [1:09:47<00:00,  1.77it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:29<00:00,  1.75it/s]
                   all       5000      36335      0.642      0.523      0.548      0.332

      
      20/99      11.1G    0.04658     0.0601    0.01583        205        640: 100%|██████████| 7393/7393 [1:06:04<00:00,  1.86it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [01:23<00:00,  1.88it/s]
                   all       5000      36335      0.638      0.517       0.55      0.335
```

### Experiment 6: Excluding the last 1x1 Conv

- Has exploding gradient issues.

```
model.model[24].qconfig = None                                          
```

```
python train.py --data coco.yaml --epochs 20 --cfg models/yolov5m.yaml \
--weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.qat.yaml \
--batch-size 32 --qat --device 2
```

Result log: 

```
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       0/19      13.7G    0.03882    0.05741    0.01344        199        640: 100%|██████████| 3697/3697 [37:28<00:00,  1.64it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:02<00:00,  1.27it/s]
                   all       5000      36335      0.706      0.557      0.609      0.415
                   
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/19      23.3G     0.0386    0.05717    0.01315        169        640: 100%|██████████| 3697/3697 [36:30<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.712      0.549      0.608      0.417

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       2/19      23.3G    0.03856     0.0571      0.013        294        640:  64%|██████▍   | 2361/3697 [23:18<13:12,  1.69it/s]train.py:394: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       2/19      23.3G    0.03851    0.05705    0.01294        144        640: 100%|██████████| 3697/3697 [36:32<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.705      0.556      0.608      0.416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       3/19      23.3G    0.03836    0.05697    0.01286        149        640: 100%|██████████| 3697/3697 [36:33<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.709      0.553      0.609      0.416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       4/19      23.3G    0.03834    0.05686    0.01285        197        640: 100%|██████████| 3697/3697 [36:30<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.711      0.559       0.61      0.418

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       5/19      23.3G    0.03832    0.05681     0.0128        110        640: 100%|██████████| 3697/3697 [36:33<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.699      0.564       0.61      0.418

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       6/19      23.3G    0.03837     0.0567    0.01279        192        640: 100%|██████████| 3697/3697 [36:31<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.701      0.556      0.608      0.416

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       7/19      23.3G     0.0383    0.05666    0.01273        134        640: 100%|██████████| 3697/3697 [36:37<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.702      0.556      0.609      0.417

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size       
       9/19      23.3G    0.03821    0.05658    0.01266        240        640: 100%|██████████| 3697/3697 [36:38<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.715      0.552      0.611      0.418
                   
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size      
      10/19      23.3G    0.03817    0.05656     0.0126        180        640: 100%|██████████| 3697/3697 [36:31<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.704      0.557      0.609      0.417

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      11/19      23.3G    0.03817    0.05667    0.01264        162        640: 100%|██████████| 3697/3697 [36:34<00:00,  1.68it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.711      0.553      0.611      0.418

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size      
      12/19      23.3G    0.03815    0.05658    0.01259        182        640: 100%|██████████| 3697/3697 [36:27<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335       0.71      0.556      0.611      0.419

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      13/19      23.3G    0.03811    0.05629    0.01251        198        640: 100%|██████████| 3697/3697 [36:29<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.708      0.557      0.611      0.419

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      14/19      23.3G    0.03803    0.05646    0.01255        237        640: 100%|██████████| 3697/3697 [36:27<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.712      0.555      0.611      0.418

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      15/19      23.3G     0.0381    0.05634    0.01256        183        640: 100%|██████████| 3697/3697 [36:31<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.705       0.56       0.61      0.419

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      16/19      23.3G    0.03801    0.05633    0.01251        228        640: 100%|██████████| 3697/3697 [36:23<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.706       0.56      0.612      0.419
                   
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size      
      17/19      23.3G      0.038    0.05623    0.01247        176        640: 100%|██████████| 3697/3697 [36:30<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335       0.71      0.558       0.61       0.42

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      18/19      23.3G    0.03796    0.05618    0.01245        216        640: 100%|██████████| 3697/3697 [36:25<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.30it/s]
                   all       5000      36335      0.703      0.562       0.61      0.419

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/19      23.3G    0.03796    0.05622    0.01243        198        640: 100%|██████████| 3697/3697 [36:29<00:00,  1.69it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:00<00:00,  1.31it/s]
                   all       5000      36335      0.705      0.558      0.611       0.42
```

### Experiment 7: FixedQParamsFakeQuantize on the last 1x1 Conv (ReLU6)

- Has exploding gradient issues.

```
bst_activation_quant_fixed = quantizer.FixedQParamsFakeQuantize.with_args(
            observer=quantizer.FixedQParamsObserver.with_args(scale=16.0/128.0, zero_point=0, dtype=torch.qint8, 
            qscheme=torch.per_tensor_affine, quant_min=-128, quant_max=127),
            quant_min=-128, quant_max=127)
```

```
# The last 1x1 Conv will use the fixed range quantization
model.model[24].qconfig = quantizer.QConfig(activation=bst_activation_quant_fixed, weight=bst_weight_quant)            
```

```
python train.py --data coco.yaml --epochs 20 --cfg models/yolov5m.yaml \
--weights runs/train/relu6/weights/best.pt --hyp data/hyps/hyp.qat.yaml \
--batch-size 32 --qat --device 2
```

Result log: 

```
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       
       1/19      15.3G     0.0397    0.05654     0.0134        169        640: 100%|██████████| 3697/3697 [38:33<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.
                   all       5000      36335      0.708      0.556      0.609      0.405    

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       7/19      17.9G    0.03973    0.05626    0.01314        466        640:  82%|████████▏ | 3038/3697 [31:44<06:53,  1.59it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       7/19      17.9G    0.03973     0.0562    0.01314        134        640: 100%|██████████| 3697/3697 [38:37<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:06<00:00,  1.
                   all       5000      36335      0.711      0.552      0.608      0.404

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       8/19      17.9G    0.03959    0.05608    0.01308        302        640:  93%|█████████▎| 3421/3697 [35:40<02:52,  1.60it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       8/19      17.9G    0.03959    0.05607    0.01308        217        640: 100%|██████████| 3697/3697 [38:32<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.
                   all       5000      36335      0.699      0.557      0.608        0.4

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       9/19      17.9G    0.03964    0.05613    0.01308        240        640: 100%|██████████| 3697/3697 [38:35<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:06<00:00,  1.
                   all       5000      36335      0.701      0.558      0.608      0.406

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      15/19      17.9G    0.03966    0.05574    0.01304        416        640:  11%|█         | 390/3697 [04:04<34:11,  1.61it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
      15/19      17.9G    0.03958    0.05595      0.013        183        640: 100%|██████████| 3697/3697 [38:33<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.
                   all       5000      36335      0.707      0.559      0.609      0.405
            
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/19      17.9G    0.03946    0.05604    0.01285        392        640:  26%|██▌       | 946/3697 [09:52<28:39,  1.60it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
      19/19      17.9G    0.03943    0.05579    0.01287        198        640: 100%|██████████| 3697/3697 [38:33<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:10<00:00,  1.12it/s]
                   all       5000      36335      0.705      0.559       0.61      0.406
```

### Experiment 8: FixedQParamsFakeQuantize on the last 1x1 Conv (ReLU)

- Has exploding gradient issues.

```
bst_activation_quant_fixed = quantizer.FixedQParamsFakeQuantize.with_args(
            observer=quantizer.FixedQParamsObserver.with_args(scale=16.0/128.0, zero_point=0, dtype=torch.qint8, 
            qscheme=torch.per_tensor_affine, quant_min=-128, quant_max=127),
            quant_min=-128, quant_max=127)
```

```
# The last 1x1 Conv will use the fixed range quantization
model.model[24].qconfig = quantizer.QConfig(activation=bst_activation_quant_fixed, weight=bst_weight_quant)            
```

```
python train.py --data coco.yaml --epochs 20 --cfg models/yolov5m.yaml \
--weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.qat.yaml \
--batch-size 32 --qat --device 2
```

Result log: 

```
Starting training for 20 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/19      12.8G    0.03478    0.05438    0.01325        385        640:   0%|          | 0/3697 [00:00<?, ?it/s]WARNING ⚠️ TensorBoard graph visualization failure 
Return value was annotated as having type Tuple[int, int] but is actually of type Tuple[NoneType, NoneType]:
  File "/bsnn/users/hongbing/Projects/bstnnx_training/bstnnx_training/PyTorch/QAT/core/observer/observer_base.py", line 128
        if self.has_customized_qrange:
            # TODO: clean this up later
            return self.quant_min, self.quant_max
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
            # # This initialization here is to be resolve TorchScript compilation issues and allow
            # # using of refinement to decouple initial_qmin and initial_qmax from quantization range.

       0/19      14.7G    0.03967     0.0565    0.01282        199        640: 100%|██████████| 3697/3697 [39:27<00:00,  1.56it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:07<00:00,  1.18it/s]
                   all       5000      36335      0.704      0.557      0.609      0.405

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/19      16.8G    0.03961    0.05632    0.01272        318        640:  22%|██▏       | 806/3697 [08:22<30:01,  1.60it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
       1/19      16.8G    0.03979    0.05644    0.01269        169        640: 100%|██████████| 3697/3697 [38:29<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.21it/s]
                   all       5000      36335      0.722       0.55      0.612      0.408

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      17/19      16.8G    0.03952    0.05564    0.01222        334        640:  73%|███████▎  | 2716/3697 [28:19<10:14,  1.60it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
      17/19      16.8G    0.03952    0.05569    0.01222        176        640: 100%|██████████| 3697/3697 [38:32<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.21it/s]
                   all       5000      36335      0.703      0.562      0.612      0.407

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      18/19      16.8G     0.0395    0.05566    0.01222        541        640:  92%|█████████▏| 3407/3697 [35:33<03:01,  1.60it/s]train.py:406: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
      18/19      16.8G     0.0395    0.05566    0.01222        216        640: 100%|██████████| 3697/3697 [38:34<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.20it/s]
                   all       5000      36335      0.701      0.562      0.612      0.407

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      19/19      16.8G    0.03949    0.05567    0.01218        198        640: 100%|██████████| 3697/3697 [38:32<00:00,  1.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [01:05<00:00,  1.20it/s]
                   all       5000      36335      0.709      0.558      0.612      0.405
```
