cd /opt/StorageArray2/workplace/xwwu-det/workplace/DeepFace/SSDFace/
./build/tools/caffe train \
--solver="models/VGGNet/WiderFace/S3FD_640x640_focal/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 4,5 2>&1 | tee jobs/VGGNet/WiderFace/S3FD_640x640_focal/VGG_WiderFace_S3FD_640x640_focal.log


#VGG_WIDER_FACE_S3FD-face_iter_30000.solverstate
#--snapshot="models/VGGNet/WiderFace/S3FD_640x640_focal/VGG_WIDER_FACE_S3FD-face_iter_30000.solverstate" \
