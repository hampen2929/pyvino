# Overview
This is the python implementation of OpenVINO models.

[Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit)

# install OpenVINO

## macOS
You can download and install from this page.

[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)

On 2019 R2, python3.7 doesn't work somehow.
need to copy ie_api.so from python3.6.
Below code solved it.

```
cp /opt/intel/openvino/python/python3.6/openvino/inference_engine/ie_api.so /opt/intel/openvino/python/python3.7/openvino/inference_engine/ie_api.so
```

## windows
[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

# setup

## macOS
Activate environment variables.
```
source /opt/intel/openvino/bin/setupvars.sh
```
## windows
Activate environment variables. 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

# Notebook samples


 # directory sturucture

```

├── pyvino
|   ├── model
|   |   └── model.py
|   ├── detector
|   |   ├── detector.py
|   |   └── detector_human.py
|   ├── segmentor
|   |   ├── segmentor.py
|   |   └── visualizer.py
|   ├── util
|   |   └── config.py
|   └── tests
└── config.ini

```


# reference
[OpenCV python demos](https://github.com/opencv/open_model_zoo/tree/master/demos/python_demos)

[OpenCV dnn](https://github.com/opencv/opencv/tree/master/samples/dnn)
