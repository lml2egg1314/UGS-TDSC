# UGS-TDSC
Source code for JPEG steganographic method of paper "Adversarial Steganography Embedding via Stego Generation and Selection"
所提方法在空域隐写与JPEG域隐写中的区别为：空域隐写中生成的候选载密图像共（N=100）+1=101个；JPEG域隐写中生成的候选载密图像共（N=1）+1=2个

* 首先，运行data/get_stego.sh文件，对载体图像集中的给定载体图像生成对应空域原始隐写方法（J-UNIWARD或者UERD）下的载密图像和嵌入代价。
* 其次，运行step2.py文件预训练目标隐写分析器。
* 然后，运行get_gradient/step3.py文件获取载体图像概率图。
* 之后，运行generate_stego/gen_filter_sets.m文件生成自适应高通核集；运行get_parameter/gen_stego.py生成候选载密图像，并计算其能否欺骗目标网络，以及其与载体图像之间的高通残差距离。
* 最后，运行generate_stego/test_UGS.m文件生成对抗载密图像。

