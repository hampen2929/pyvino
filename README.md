# install openvino

## macOS

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html

python3.7 doesn't work.
need to copy ie_api.so from python3.6.

cp /opt/intel/openvino_2019.2.242/python/python3.6/openvino/inference_engine/ie_api.so /opt/intel/openvino_2019.2.242/python/python3.7/openvino/inference_engine/

# setup

## macOS
```
source /opt/intel/openvino/bin/setupvars.sh
```
 ## windows
 ```
conda activate openvino
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
cd \Users\yuya.mochimaru\Desktop\openvino-python
```
 # directory sturucture

```

├── vinopy
|   ├── detector
|   |   └── detector.py
|   ├── model
|   |   ├── model.py
|   |   ├── model_face.py
|   |   └── model_body.py
|   ├── util
|   |   └── config.py
|   └── tests
└── config.ini

```

# pytest

```
pytest vinopy
```

# reference

https://gist.github.com/kodamap/aa747ae2058bbb919e0308cf8b5f1718
