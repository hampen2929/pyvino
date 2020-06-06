# First time
## Clone repository
```
mkdir $HOME/workspace
cd $HOME/workspace
git clone https://github.com/hampen2929/pyvino.git
cd pyvino
```

## Build docker image

```
docker build -t pyvino_image .
```

## Run container
### CPU
```
docker run -it \
--name pyvino \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/workspace:/workspace \
-p 8888:8888 \
pyvino_image_dev \
/bin/bash
```

## Setup pyvino
```
cd /workspace/pyvino
python setup.py develop
```

## Test
```
pytest pyvino
```

Test result images are here.
```
pyvino/pyvino/tests/data
```

# Notebook

```
jupyter notebook
``` 
Notebook samples are [HERE](https://github.com/hampen2929/pyvino/tree/master/notebook).

# From the second time
Insted of build and run, use below.
## Start
```
docker start pyvino
```

## Execute
```
docker exec -it pyvino bash
```

export MO_ROOT="/opt/intel/openvino/deployment_tools/model_optimizer"

python $MO_ROOT/mo_tf.py \
--input_model /workspace/pyvino/temp/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb \
--transformations_config $MO_ROOT/extensions/front/tf/yolo_v3.json \
--batch 1

