# install 

## macOS
### OpenVINO

You can download and install from this page.

[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)

On 2019 R2, python3.7 doesn't work somehow.
need to copy ie_api.so from python3.6.
Below code solved it.

```
cp /opt/intel/openvino/python/python3.6/openvino/inference_engine/ie_api.so /opt/intel/openvino/python/python3.7/openvino/inference_engine/ie_api.so
```

### envs

Create conda env
```
conda create -n pyvino python==3.6.5
source activate pyvino
ipython kernel install --user --name=jupytepyvino --display-name=pyvino 
```

Activate environment variables.
```
source /opt/intel/openvino/bin/setupvars.sh
```


## windows
[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

## windows
Activate environment variables. 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```
