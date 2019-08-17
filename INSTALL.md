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
```buildoutcfg
conda create -n pyvino python==3.6.5
source activate pyvino 
```

install pyvino
```buildoutcfg
python setup.py install
```

Activate environment variables.
```buildoutcfg
source /opt/intel/openvino/bin/setupvars.sh
```

### jupyter
if you want to use jupyter notebook with conda env, please refer below.
if not, please skip.

install jupyter_environment_kernels
```buildoutcfg
pip install environment_kernels
```

generate jupyter config
```buildoutcfg
jupyter notebook --generate-config
```
add below to ~/.jupyter/jupyter_notebook_config.py 
```buildoutcfg
c.NotebookApp.kernel_spec_manager_class = 'environment_kernels.EnvironmentKernelSpecManager'
c.EnvironmentKernelSpecManager.env_dirs = ['/Users/username/anaconda/envs/']
```

start
```buildoutcfg
jupyter notebook
```

### Notebook example
Notebook examples is [HERE]()


## windows
[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

## windows
Activate environment variables. 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```
