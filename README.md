# Rot-E ASISR
Code of "Rotation Equivariant Arbitrary-scale Image Super-Resolution"  

    configs\             : Folder for Storing Configuration Files
    datasets\            : Codes for reading samples for different datasets
    exampleImage\        : An example image for conducting equivariance observation experiments
    models\              : Code of different rotataion equivariant encoders and INRs
    EQ_Observe.py        : Code for conducting equivariance observation experiments
    test.py              : Code for testing the trained models 
    test_swin.py         : Code for testing the trained models whose encoder is swinIR method 
    train.py             : Code for training the ASISR models
    utils.py             : Code of functions that may be utilized
    
    
We make efforts to construct a rotation equivariant ASISR method in this study. Specifically, we elaborately redesign basic architectures of INR and encoder modules, incorporating intrinsic rotation equivariance capabilities beyond those of conventional ASISR networks. Through such amelioration, the
ASISR task can, for the first time, be implemented with end-to-end rotational equivariance maintained from input to output throughout the network flow.

![Illustration of EQ-ASISR](https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/Fig2.png)
Figure 1. Illustration of overall rotation equivariant arbitrary-scale image super-resolution

The capability of the proposed framework in keeping the rotation symmetry can be observed from the following figures. 

![Illustration of EQ LIIF](https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/Liif_2_iteration.gif)
Figure 2. Illustration of the output local implicit image function obtained by LIIF and LIIF enhanced with the proposed method (LIIF-EQ) with different training epoches. It can be observed that the proposed LIFF-EQ consistently maintains the rotational symmetry characteristics of the data across different training epoches, whereas the original LIIF method does not.

<img src="https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/Liif_rotaion.gif">
Figure 2. Illustration of the output local implicit image function obtained by LIIF and LIIF enhanced with the proposed method (LIIF-EQ) with 20 training epoches. It can be observed that the implicit function obtained by the proposed LIIF-EQ can stably rotate with the rotation of the input, whereas LIIF cannot.


.

**Usage:**    

Examples for training the proposed methods:

    # Train LIIF and LIIF-EQ
    python train.py --config configs/train-div2k/train-edsr-baseline-liif.yaml
    python train.py --config configs/train-div2k/train-edsr-baseline-liif-EQ.yaml

    # Train OPE and OPE-EQ
    python train.py --config configs/train-div2k/train-edsr-baseline-ope.yaml
    python train.py --config configs/train-div2k/train-edsr-baseline-ope-EQ.yaml

    # Train LTE and LTE-EQ
    python train.py --config configs/train-div2k/train-edsr-baseline-lte.yaml
    python train.py --config configs/train-div2k/train-edsr-baseline-lte-EQ.yaml

Examples for conducting equivariance observation experiments

    # Observe the equivariance of LIIF and LIIF-EQ
    python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-liif.yaml
    python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-liif-EQ.yaml

    # Observe the equivariance of OPE and OPE-EQ
    python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-ope.yaml
    python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-ope-EQ.yaml

    # Observe the equivariance of LTE and LTE-EQ
    python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-lte.yaml
    python EQ_Observe.py  --config configs/observation/Observe-edsr-baseline-lte-EQ.yaml

    # Observe the equivariance of swinIR based LIIF and LIIF-EQ
    python EQ_Observe.py  --config configs/observation/Observe-swinir-lte.yaml
    python EQ_Observe.py  --config configs/observation/Observe-swinir-lte-EQ.yaml

The output image of equivariance observation experiments would be like:
<img src="https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/EqExample_lte.png">
Figure 4. Illustration of the output images of original and its rotation-equivariant improvement (p16 rotation equivariant), when the network is randomly initialized with out any training.

    
