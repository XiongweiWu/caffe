cd /opt/StorageArray2/workplace/xwwu-det/workplace/DeepFace/SSDFace/
./build/tools/caffe train \
--solver="models/VGGNet/WiderFace/S3FD_640x640_ohem/solver.prototxt" \
--snapshot="models/VGGNet/WiderFace/S3FD_640x640_ohem/VGG_WIDER_FACE_S3FD-face_iter_30000.solverstate" \
--gpu 4,6 2>&1 | tee jobs/VGGNet/WiderFace/S3FD_640x640_ohem/VGG_WiderFace_S3FD_640x640_ohem.log


#VGG_WIDER_FACE_S3FD-face_iter_30000.solverstate
#--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
