# EQ-ASISR
Code of "Rotation Equivariant Arbitrary-scale Image Super-Resolution"  

![Illustration of EQ-ASISR](https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/Fig2.png)
Figure 1. Illustration of overall rotation equivariant arbitrary-scale image super-resolution


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

The capability of the proposed framework in keeping the rotation symmetry can be observed from the following figures. 

![Illustration of EQ LIIF](https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/Liif_2_iteration.gif)
Figure 2. Illustration of the output local implicit image function obtained by LIIF and LIIF enhanced with the proposed method (LIIF-EQ)

<img src="https://github.com/XieQi2015/ImageFolder/blob/master/EQ-ASISR/Liif_rotaion.gif">
