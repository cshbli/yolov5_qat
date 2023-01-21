# Yolov5 Quantization Aware Training QAT

## Notes

**This repo is based on the release v7.0 of [yolov5](https://github.com/ultralytics/yolov5/).**

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

### 1.4 Replacing SiLU with ReLU (Optional)

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

Here is the complete retraining log file [retraining after replacing SiLU with ReLU](./relu_retraining.csv).

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

## Post Training Quantization

```
python train.py --data coco.yaml --epochs 20 --cfg models/yolov5m.yaml --weights runs/train/relu/weights/best.pt --hyp data/hyps/hyp.qat.yaml --batch-size 32 --ptq --device 0
```

Post training quantization results:

```
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [23:23<00:00, 17.77s/it]
                   all       5000      36335      0.699       0.57      0.619      0.431
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [06:00<00:00,  4.56s/it]
                   all       5000      36335      0.699      0.563      0.615        0.4
```

```
Plotting labels to runs/train/exp4/labels.jpg... 
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [23:23<00:00, 17.76s/it]
                   all       5000      36335      0.699       0.57      0.619      0.431
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 79/79 [05:57<00:00,  4.53s/it]
                   all       5000      36335      0.696      0.558       0.61      0.401
                person       5000      10777      0.781      0.714      0.786        0.5
               bicycle       5000        314      0.674      0.503      0.575       0.31
                   car       5000       1918      0.711      0.612      0.674      0.415
            motorcycle       5000        367      0.747       0.66      0.728      0.438
              airplane       5000        143      0.799      0.811      0.886      0.612
                   bus       5000        283      0.844      0.746      0.822       0.61
                 train       5000        190      0.875       0.81      0.886      0.626
                 truck       5000        414      0.623      0.507      0.566      0.357
                  boat       5000        424      0.678      0.425      0.519      0.256
         traffic light       5000        634      0.672      0.498      0.544      0.265
          fire hydrant       5000        101      0.938      0.792      0.865      0.652
             stop sign       5000         75      0.818      0.658      0.766       0.61
         parking meter       5000         60      0.807        0.6      0.641      0.466
                 bench       5000        411      0.601      0.338       0.38      0.233
                  bird       5000        427      0.694      0.459      0.532      0.343
                   cat       5000        202      0.873      0.785       0.86       0.63
                   dog       5000        218      0.773      0.734      0.792      0.601
                 horse       5000        272      0.806      0.739      0.831      0.571
                 sheep       5000        354      0.675       0.74      0.767      0.514
                   cow       5000        372      0.762      0.726      0.797      0.528
              elephant       5000        252      0.781      0.869       0.85      0.598
                  bear       5000         71       0.88      0.859       0.89      0.703
                 zebra       5000        266      0.879      0.831      0.906      0.655
               giraffe       5000        232      0.863      0.867      0.924      0.681
              backpack       5000        371       0.53      0.267      0.305       0.16
              umbrella       5000        407      0.702      0.624      0.654      0.412
               handbag       5000        540      0.536       0.25      0.282      0.154
                   tie       5000        252      0.724      0.541      0.568      0.315
              suitcase       5000        299      0.655      0.545      0.608      0.383
               frisbee       5000        115      0.816      0.826      0.874       0.62
                  skis       5000        241      0.637      0.386      0.454      0.225
             snowboard       5000         69      0.641      0.449      0.467      0.298
           sports ball       5000        260      0.748      0.583      0.651      0.412
                  kite       5000        327      0.652      0.566      0.622      0.403
          baseball bat       5000        145      0.743      0.566      0.613      0.341
        baseball glove       5000        148      0.777      0.568       0.63      0.357
            skateboard       5000        179      0.777       0.76      0.786      0.533
             surfboard       5000        267      0.751      0.539      0.624      0.362
         tennis racket       5000        225        0.8      0.764      0.805      0.496
                bottle       5000       1013      0.641      0.493      0.558      0.352
            wine glass       5000        341      0.681      0.504       0.57      0.337
                   cup       5000        895      0.671      0.552       0.61      0.409
                  fork       5000        215      0.663      0.431      0.527      0.341
                 knife       5000        325      0.558      0.256      0.325      0.191
                 spoon       5000        253      0.566      0.281      0.336      0.197
                  bowl       5000        623      0.617      0.507      0.569      0.373
                banana       5000        370      0.491       0.33      0.356      0.204
                 apple       5000        236      0.458      0.294      0.274      0.183
              sandwich       5000        177      0.581      0.469      0.519      0.377
                orange       5000        285      0.495      0.386      0.382      0.269
              broccoli       5000        312      0.548      0.392      0.417      0.218
                carrot       5000        365      0.414      0.323      0.289      0.172
               hot dog       5000        125      0.724      0.472      0.552      0.362
                 pizza       5000        284      0.764      0.669      0.734      0.502
                 donut       5000        328      0.597      0.578      0.621      0.449
                  cake       5000        310      0.626      0.556      0.606      0.364
                 chair       5000       1771      0.624       0.43      0.498      0.293
                 couch       5000        261      0.733      0.556      0.645      0.442
          potted plant       5000        342      0.579       0.43      0.455       0.25
                   bed       5000        163      0.724      0.521      0.642       0.42
          dining table       5000        695      0.599      0.397      0.428      0.272
                toilet       5000        179       0.78      0.777      0.851      0.622
                    tv       5000        288      0.829      0.726      0.805      0.563
                laptop       5000        231      0.812      0.697      0.756       0.57
                 mouse       5000        106      0.756      0.755      0.768       0.54
                remote       5000        283      0.585      0.459      0.496      0.282
              keyboard       5000        153      0.712      0.646      0.708      0.489
            cell phone       5000        262      0.639      0.546      0.571      0.351
             microwave       5000         55       0.72       0.75        0.8      0.587
                  oven       5000        143      0.642       0.49      0.568      0.345
               toaster       5000          9      0.464      0.444      0.399      0.209
                  sink       5000        225      0.687       0.56      0.588      0.369
          refrigerator       5000        126      0.775      0.667      0.738      0.524
                  book       5000       1129      0.475      0.178      0.237      0.112
                 clock       5000        267      0.793        0.7      0.738       0.47
                  vase       5000        274      0.594      0.534      0.559      0.352
              scissors       5000         36        0.8      0.333      0.412      0.276
            teddy bear       5000        190      0.706      0.619      0.678      0.454
            hair drier       5000         11          1          0     0.0967     0.0678
            toothbrush       5000         57      0.518      0.421      0.407      0.241

Evaluating pycocotools mAP... saving runs/train/exp4/_predictions.json...
loading annotations into memory...
Done (t=0.30s)
creating index...
index created!
Loading and preparing results...
DONE (t=3.87s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=43.25s).
Accumulating evaluation results...
DONE (t=8.79s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.637
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.740
```