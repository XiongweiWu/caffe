cd /home/xwwu/workplace/CVPR2018/DeepFace/SSDFace
./build/tools/caffe train \
--solver="models/VGGNet/WiderFace/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 4,5,6,7 2>&1 | tee jobs/VGGNet/WiderFace/SSD_300x300/VGG_WiderFace_SSD_640x640.log
