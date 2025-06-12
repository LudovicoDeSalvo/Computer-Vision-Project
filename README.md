# License Plate Recognition on CCPD2019

This repository implements a complete pipeline for license plate detection and recognition using the CCPD2019dataset. It combines **YOLOv5** for plate detection and two recognition backends:
- A **CNN + LSTM + CTC** baseline.
- The **PDLPR** model (Parallel Decoder License Plate Recognition), a transformer-based architecture designed for fast, high-accuracy recognition without character segmentation.


## Project Structure

```
├── main.py     # Full interactive pipeline: from raw data to final inference
├── baseline/   # Optional CNN+LSTM+CTC baseline model
│ ├── model.py  
│ ├── train.py  
│ ├── eval.py   
├── data/
│ └── CCPD2019/         # Raw CCPD images
│ └── CCPD2019_crops/   # YOLO-based plate crops (48x144)
├── pdlpr/
│ ├── model.py 
│ ├── train.py 
│ ├── eval.py 
├── utils/
│ ├── crop_all.py                      # Calls crop.py for each sub-dataset
│ ├── crop.py                          # Crops plates from images using YOLO labels
│ ├── generate_yolo_labels.py          
│ ├── train_yolo.py                 
│ └── base_splitter.py                 # Divedes the ccpd_base into train and validaiton (90%-10%)
└── yolov5/             # YOLOv5 repo (cloned)
```

## How to Use `main.py`

Run the full pipeline interactively, **recommended**:

```bash
python main.py
```

You will be prompted for each step: download, train YOLO, crop plates, select recognition model, train it, and run inference.

Alternatively, run in scripted mode with arguments:

### ⚙️ `main.py` Arguments

| Argument               | Type      | Description |
|------------------------|-----------|-------------|
| `--non_interactive`    | flag      | Run without prompts, useful for scripting full pipeline. |
| `--clone_yolo`         | flag      | Clone the YOLOv5 repository into `yolov5/`. |
| `--download`           | flag      | Download and extract the CCPD2019 dataset (tar.gz + nested .tar). |
| `--train`              | flag      | Train YOLOv5 on CCPD2019. |
| `--gen_labels`         | flag      | Generate YOLOv5-compatible labels from CCPD filenames. |
| `--crop`               | flag      | Crop license plate regions to 48×144 using YOLO predictions. |
| `--model`              | string    | Select recognition model: `baseline` or `pdlpr`. Required for non-interactive mode. |
| `--subset`             | string    | Choose CCPD subset for inference. Options:<br>`ccpd_base`, `ccpd_blur`, `ccpd_challenge`, `ccpd_db`, `ccpd_fn`, `ccpd_np`, `ccpd_rotate`, `ccpd_tilt`, `ccpd_weather`. |
| `--crop_dir`           | path      | Manually specify a directory of 48×144 cropped plates (overrides default). |
| `--skip_infer`         | flag      | Skip the final inference step. Useful if you only want preprocessing/training. |

## References

[Reference Paper](https://www.mdpi.com/1424-8220/24/9/2791)
A Real-Time License Plate Detection and Recognition Model in Unconstrained Scenarios 
-Lingbing Tao, Shunhe Hong, Yongxing Lin, Yangbing Chen, Pingan He, Zhixin Tie 

[CCPD2019 Dataset](https://github.com/detectRecog/CCPD)
Xu et al., 2018. ECCV

[YOLOv5](https://github.com/ultralytics/yolov5)