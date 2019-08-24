# System requirements

## Processors
6th to 9th generation Intel® Core™ and Intel® Xeon® processors

## Compatible Operating Systems
- Ubuntu* 16.04.3 LTS (64 bit)
- Windows® 10 (64 bit)
- macOS* 10.13, 10.14 (64 bit)

# Installation

## Create and activate conda env
```buildoutcfg
conda create -n pyvino python==3.6.5 -y
source activate pyvino
```

## OpenVINO
### Download and install
Download link

- [MacOS](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)

- [WIndows10](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

- [Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

pyvino supports only ver 2019 R2.

### Set up variables

Before use openvino, variables setup is needed every time.

MacOS/Linux
```buildoutcfg
source /opt/intel/openvino/bin/setupvars.sh
```
Add `source /opt/intel/openvino/bin/setupvars.sh` to `~/.bashrc` is recommended.

Windows10
```buildoutcfg
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

### clone repository

Move to directory for clone
```buildoutcfg
cd ~
```
clone

```buildoutcfg
git clone https://github.com/hampen2929/pyvino.git
cd pyvino
``` 

install pyvino
```buildoutcfg
python setup.py install
```

## sample script
Test to confirm whether the installation is success.
```buildoutcfg
python test_script.py
```
If success, this image appears.

![image](https://user-images.githubusercontent.com/34574033/63309083-657c4400-c330-11e9-8b72-754ab8ba9cce.png)

Click "q" to exit.

## Notebook samples
Notebook samples are [HERE](https://github.com/hampen2929/pyvino/blob/master/notebook/).
