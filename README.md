## Code for CVPR2020 Paper "Diverse Slow-Drop Block Multi-Branch Network for Person Re-Identiﬁcation"

This code is developed based on pytorch framework.

### Prerequisites

- Pytorch(1.1.0)
- yacs (0.1.6) 
- torchvision(0.3.0)
- tb-nightly(2.0.0)
- Cython(0.29.12)
- pytorch-ignite(0.1.2)

### SetUp

```
git clone https://github.com/freeSubmission/SDB.git
```

### Data Preparation(Duke)

- dataset
  - dukemtmc-reid
    - DukeMTMC-reID
      - bounding_box_train
      - bounding_box_test
      - query

### Train

```
cd SDB/scripts
vim train.sh
bash train.sh
```

### Results(no rerank)

|                   | rank 1 | mAP  |
| :---------------: | :----: | :--: |
|  **MARKET 1501**  |  95.9  | 88.7 |
|     **DUKE**      |  90.2  | 79.5 |
| **CUHK-labeled**  |  80.8  | 78.7 |
| **CUHK-detected** |  77.9  | 75.2 |

### Datasets

- Market1501 <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`
- `CUHK03 <https://www.cvfoundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`
- `DukeMTMC-reID <https://arxiv.org/abs/1701.07717>`





