# install 

## macOS
### OpenVINO

You can download and install from this page.

[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)

On 2019 R2, python3.7 doesn't work somehow.
need to copy ie_api.so from python3.6.
Below command solved it.

```
cp /opt/intel/openvino/python/python3.6/openvino/inference_engine/ie_api.so /opt/intel/openvino/python/python3.7/openvino/inference_engine/ie_api.so
```

### pyvino

Create conda env
```buildoutcfg
conda create -n pyvino python==3.6.5
source activate pyvino 
```

Activate environment variables.
```buildoutcfg
source /opt/intel/openvino/bin/setupvars.sh
```

clone repository
```buildoutcfg
git clone https://github.com/hampen2929/pyvino.git
cd pyvino
``` 

install pyvino
```buildoutcfg
python setup.py install
```

### edit config
Set cpu extension path.
```buildoutcfg
vi config.ini
```
The location of cpu extension is depends on OpenVION version.

In case of 2019 R2, the path is `/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib`

### Set config.ini and intel_models
The file config.ini and intel_models need to be located at the under home directory. 
```buildoutcfg
mkdir ~/.pyvino
mv config.ini ~/.pyvino
mv intel_models ~/.pyvino/intel_models
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

Edit `jupyter_notebook_config.py`
```buildoutcfg
vi ~/.jupyter/jupyter_notebook_config.py
``` 
add below to the end of `jupyter_notebook_config.py`.

If you use miniconda, set proper path. 
```buildoutcfg
c.NotebookApp.kernel_spec_manager_class = 'environment_kernels.EnvironmentKernelSpecManager'
c.EnvironmentKernelSpecManager.env_dirs = ['/Users/username/anaconda3/envs/']
```

execute jupyter notebook
```buildoutcfg
jupyter notebook
```

### Notebook example
Notebook examples are [HERE](https://github.com/hampen2929/pyvino/blob/master/notebook/).


## windows
[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

Activate environment variables. 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

TBD
