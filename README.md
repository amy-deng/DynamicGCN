## DynamicGCN
This is the source code for paper [Learning Dynamic Context Graphs for Predicting Social Events](https://dl.acm.org/citation.cfm?id=3330919 "Learning Dynamic Context Graphs for Predicting Social Events") appeared in KDD2019 (research track)

[Songgaojun Deng](https://amy-deng.github.io/home/), [Huzefa Rangwala](https://cs.gmu.edu/~hrangwal/), [Yue Ning](https://yue-ning.github.io/)

### Data
- [ICEWS event data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075 "ICEWS event data") is available online.
- ICEWS news data has not been released publicly.（If you want to access the original news text information of the event, I suggest [GDELT data](https://www.gdeltproject.org/ "Link").）

### Libraries
- **PyTorch >= 1.0**
- **sklearn**
- **pytorch_sparse** Refer to [the official website](https://github.com/rusty1s/pytorch_sparse "this page") to install.

### Sample dataset
- **THAD6h** (Thailand dynamic (temporal) dataset, around 600 nodes per graph) [Google Drive](https://drive.google.com/drive/folders/11fuAkgybqPOEirQFQi8wkld_R5dZkv74?usp=sharing "Google Drive")
- **INDD6h** [Google Drive](https://drive.google.com/drive/folders/1Yvo_sgWnWaT90nL1MzELY46z0IqoFE7S?usp=sharing "Google Drive")
- **EGYD6h** [Google Drive](https://drive.google.com/drive/folders/1lUk8VvAKDslObDUdVE3sf0cyDp5oKI8l?usp=sharing "Google Drive")
- **RUSD6h** [Google Drive](https://drive.google.com/drive/folders/1aEDqijSYTrjNNAfgHCNe37slwmuJL4Ay?usp=sharing "Google Drive")
  - **\*.idx** / **\*.tidx**      Word index file for training/testing
  - **\*.x** / **\*.tx**      Temporal graph input file for training/testing
  - **\*.y** / **\*.ty**      Ground truth for training/testing


## Cite

Please cite our paper if you find this code useful for your research:

```
@inproceedings{deng2019learning,
  title={Learning Dynamic Context Graphs for Predicting Social Events},
  author={Deng, Songgaojun and Rangwala, Huzefa and Ning, Yue},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1007--1016},
  year={2019}
}
```
