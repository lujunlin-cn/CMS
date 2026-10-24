# Manual image review

- `train/subject_05239/study_13172/pic_021690.jpg`: valid JPEG and metadata dimensions, but 97.23% pixels are zero and visual review shows only a small partial radiographic region in the lower-right corner. The same Study has `pic_021691.jpg`, a complete chest radiograph with the same label. This is a low-information acquisition/cropping anomaly, not evidence of an adversarial attack.
- Original data are retained. The baseline training loader reads `audit/exclude_train_images.csv` and excludes this one image only from baseline training; validation/test are unchanged.
