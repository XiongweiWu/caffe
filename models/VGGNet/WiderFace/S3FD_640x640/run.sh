cd /opt/StorageArray2/workplace/xwwu-det/workplace/DeepFace/SSDFace/
./build/tools/caffe train \
--solver="models/VGGNet/WiderFace/S3FD_640x640/solver.prototxt" \
--snapshot="models/VGGNet/WiderFace/S3FD_640x640/VGG_WIDER_FACE_S3FD-face_iter_100000.solverstate" \
--gpu 5 2>&1 | tee jobs/VGGNet/WiderFace/S3FD_640x640/VGG_WiderFace_S3FD_640x640.log


#VGG_WIDER_FACE_S3FD-face_iter_30000.solverstate
