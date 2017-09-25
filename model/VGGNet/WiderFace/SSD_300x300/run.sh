cd /opt/StorageArray2/workplace/xwwu-det/workplace/DeepFace/SSDFace
./build/tools/caffe train \
--solver="models/VGGNet/WiderFace/SSD_300x300/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 5 2>&1 | tee jobs/VGGNet/WiderFace/SSD_300x300/VGG_WiderFace_SSD_640x640.log
