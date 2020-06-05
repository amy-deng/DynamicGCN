## DynamicGCN
Source code for paper "[Learning Dynamic Context Graphs for Predicting Social Events](https://dl.acm.org/citation.cfm?id=3330919 "Learning Dynamic Context Graphs for Predicting Social Events")".

### Data
- [ICEWS event data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075 "ICEWS event data") is available online.
- ICEWS news data has not been released publicly.

### Libraries
- **PyTorch >= 1.0**
- **sklearn**
- **pytorch_sparse** Refer to [the official website](https://github.com/rusty1s/pytorch_sparse "this page") to install.

### Sample dataset
- **THAD6h** (Thailand dynamic (temporal) dataset, around 600 nodes per graph) [Link](https://drive.google.com/open?id=1l1vBoldu1U_ktqKT8tr9HyTHDydEmXWo "Link")
- **INDD6h** [Link](https://drive.google.com/drive/folders/1ySdGDpLlBbh1XuG5cAL9FmFFVrgE7-wr?usp=sharing "Link")
- **EGYD6h** [Link](https://drive.google.com/drive/folders/1ZvVn81TZF7kn3kh9eIMlm2YX6NeV6hxG?usp=sharing "Link")
- **RUSD6h** [Link](https://drive.google.com/drive/folders/1EikE191TA7rx_YhmGsMOrjaW7fFmVpLV?usp=sharing "Link")
  - **\*.idx** / **\*.tidx**      Word index file for training/testing
  - **\*.x** / **\*.tx**      Temporal graph input file for training/testing
  - **\*.y** / **\*.ty**      Ground truth for training/testing
