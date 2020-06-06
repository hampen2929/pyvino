# Overview

This is the python implementation of OpenVINO models.

The OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that emulate human vision. 
Based on Convolutional Neural Networks (CNNs), the toolkit extends CV workloads across Intel® hardware, maximizing performance.

[Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit)

# Supported tasks

We support these tasks.

|task                    |model                                     |
|------------------------|------------------------------------|
|detect_face             |face-detection-adas-0001            |
|emotion_recognition     |emotions-recognition-retail-0003    |
|estimate_headpose       |head-pose-estimation-adas-0001      |
|detect_body             |person-detection-retail-0013        |
|estimate_humanpose      |human-pose-estimation-0001          |
|estimate_3d_humanpose   |human-pose-estimation-3d            |
|detect_segmentation     |instance-segmentation-seurity-0050  |

Computed results by models are as below.

![supported_models](https://user-images.githubusercontent.com/34574033/63226303-36bc7b80-c213-11e9-8881-74241128e1d3.png)

![image](https://user-images.githubusercontent.com/34574033/71774364-6563a480-2fb0-11ea-8d3b-37399101bc32.png)

# Installation
## Use docker
Click [HERE](https://github.com/hampen2929/pyvino/blob/master/DOCKER.md) for installation instructions using docker.

## No use docker
Click [HERE](https://github.com/hampen2929/pyvino/blob/master/INSTALL.md) for installation instructions.


# Notebook samples
Notebook samples are [HERE](https://github.com/hampen2929/pyvino/tree/master/notebook).

# Update
v0.0.1 (2019/08/24)
- Mac, Windows and Ubuntu are supported/

v0.0.2 (2019/01/04)
- 3d pose estimation, docker supported

v0.0.3 (2020/06/07)
- OpenVino 2020 R3 LTS version is based

# License
This project is released under the [Apache 2.0 license](https://github.com/hampen2929/pyvino/blob/master/LICENSE).

# Reference
[OpenCV python demos](https://github.com/opencv/open_model_zoo/tree/master/demos/python_demos)

[OpenCV dnn](https://github.com/opencv/opencv/tree/master/samples/dnn)

[MODEL_ZOO](https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/)

[3D pose estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
