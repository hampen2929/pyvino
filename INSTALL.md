# System requirements

## Processors
6th to 9th generation Intel® Core™ and Intel® Xeon® processors

## Compatible Operating Systems
- Ubuntu* 16.04.3 LTS (64 bit)
- Windows® 10 (64 bit)
- CentOS* 7.4 (64 bit)
- macOS* 10.13, 10.14 (64 bit)

# Installation

## macOS
### OpenVINO

Please refer [HERE](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html) for installation.

In my environment, version 2019 R2 of OpenVINO did not work properly.

Need to copy ie_api.so from python3.6.

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

Before use openvino, this command is needed every time.

Add `source /opt/intel/openvino/bin/setupvars.sh` to `~/.bash_profile` is recommended.

clone repository
```buildoutcfg
git clone https://github.com/hampen2929/pyvino.git
cd pyvino
``` 

install pyvino
```buildoutcfg
python setup.py install
```

### Set config file
Set cpu extension path.
```buildoutcfg
vi config.ini
```
The location of cpu extension is depends on OpenVION version.

In case of 2019 R2, the path is `/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib`

The file config.ini and intel_models need to be located at the under home directory. 
```buildoutcfg
mkdir ~/.pyvino
cp config.ini ~/.pyvino/
```

### Download intel models

We support these tasks.

|task                    |model                                     |
|------------------------|------------------------------------|
|detect_face             |face-detection-adas-0001            |
|emotion_recognition     |emotions-recognition-retail-0003    |
|estimate_headpose       |head-pose-estimation-adas-0001      |
|detect_body             |person-detection-retail-0013        |
|estimate_humanpose      |human-pose-estimation-0001          |
|detect_segmentation     |instance-segmentation-security-0050 |


Download intel_models with this command.
```buildoutcfg
python download_intel_models.py
```

Download models from [open model zoo](https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/)
and place to `~/.pyvino/intel_models/` as below.

```
HOME
└── .pyvino
    └── intel_models
        ├── face-detection-adas-0001
        |   └── FP32
        |       ├── face-detection-adas-0001.xml
        |       └── face-detection-adas-0001.bin
       ~~~
        └── instance-segmentation-security-0050
            └── FP32
                ├── instance-segmentation-security-0050.xml
                └── instance-segmentation-security-0050.bin
```

### test command
Test to confirm the installation is success or not.
```buildoutcfg
python test_script.py
```

If success, this image appears.
![image](https://user-images.githubusercontent.com/34574033/63309083-657c4400-c330-11e9-8b72-754ab8ba9cce.png)

Click "q" to exit.

## jupyter notebook
if you use jupyter notebook with conda env, please refer below.

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
You can chose pyvino env from Kernel -> Change kernel.

## Notebook samples
Notebook samples are [HERE](https://github.com/hampen2929/pyvino/blob/master/notebook/).


## windows
### OpenVINO
[install link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

### pyvino

Activate environment variables. 
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

TBD
