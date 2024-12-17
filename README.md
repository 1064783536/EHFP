# CHIFE
we propose a complementary height information enhancement framework (CHIEF) to improve the performance of voxel-based 3D object detection method.

# Installation
Install mmdetection3d and make sure it reproduces the pointpillar algorithm correctly!(https://github.com/open-mmlab/mmdetection3d/tree/main)

# Code
Just replace the data_preprocessor.py and voxelnet.py files in the mmdetection3d code
The path of file "data_preprocessor.py" is "mmdet3d/models/data_preprocessors/data_preprocessor.py"
The path of file "voxelnet.py" is "mmdet3d/models/detectors/voxelnet.py"
# Train
python tools/train.py {config}
# Test
python tools/test.py {config}
