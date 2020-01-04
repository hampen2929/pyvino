# First time
## clone repository
```
mkdir $HOME/src_dir
cd $HOME/src_dir
git clone https://github.com/hampen2929/pyvino.git
cd pyvino
```

## build

```
docker build -t pyvino_image .
```

## run
### CPU
```
docker run -it \
--name pyvino \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/src_dir:/home/ubuntu/src_dir \
-p 8888:8888 \
pyvino_image \
/bin/bash
```

## setup
```
cd $HOME/src_dir/pyvino
python setup.py install
```


<!-- ## activate 
```
source activate idp
```

## env
path to pose extractor module for 3d pose estimation
```
export PYTHONPATH=/home/ubuntu/src_dir/pyvino/pyvino/model/human_pose_estimation/human_3d_pose_estimator/pose_extractor/build/
```

setupvars for openvino
```
source /opt/intel/openvino/bin/setupvars.sh
``` -->

## test
```
pytest pyvino
```
cheack result here
```
$HOME/src_dir/pyvino/pyvino/tests/data
```

## jupyter notebook
```
jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root
``` 

# From the second time
Insted of build and run, use below.
## start
```
docker start pyvino
```

## exec
```
docker exec -it pyvino bash
```
