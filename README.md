# SVP
we propose a stratified vertical priors framework to improve the pillar-based 3D object detection algorithms.

# Installation
Install mmdetection3d and make sure it reproduces the pointpillar algorithm correctly!(https://github.com/open-mmlab/mmdetection3d/tree/main)

# Code
Just replace the data_preprocessor.py and voxelnet.py files in the mmdetection3d code
The path of file "data_preprocessor.py" is "mmdet3d/models/data_preprocessors/data_preprocessor.py"
The path of file "voxelnet.py" is "mmdet3d/models/detectors/voxelnet.py"
# Train
python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
# Test
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py
