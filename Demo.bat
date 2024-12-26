CALL conda activate XQtorch
d:
cd D:\2021_SR_Code\UnLimitedSR-EQ-Reselsed-Code

%python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-liif.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-liif-EQ.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-ope.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-ope-EQ.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-lte.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-lte-EQ.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-swinir-lte.yaml%

%python EQ_Observe.py  --config configs/observation/Observe-swinir-lte-EQ.yaml%

%python train.py --config configs/train-div2k/train-edsr-baseline-liif.yaml%

%python train.py --config configs/train-div2k/train-edsr-baseline-liif-EQ.yaml%

%python train.py --config configs/train-div2k/train-edsr-baseline-ope.yaml%

%python train.py --config configs/train-div2k/train-edsr-baseline-ope-EQ.yaml%

%python train.py --config configs/train-div2k/train-edsr-baseline-lte.yaml%

%python train.py --config configs/train-div2k/train-edsr-baseline-lte-EQ.yaml

%PAUSE