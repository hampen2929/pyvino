# First time
## Clone repository
```
mkdir $HOME/src_dir
cd $HOME/src_dir
git clone https://github.com/hampen2929/pyvino.git
cd pyvino
```

## Build docker image

```
docker build -t pyvino_image_gpu .
```

## Run container
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

### GPU
need to install nvidia-docker
```
docker run --runtime=nvidia -it \
--name pyvino_gpu \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/src_dir:/home/ubuntu/src_dir \
-p 8888:8888 \
pyvino_image_gpu \
/bin/bash
```

## Setup pyvino
```
cd /home/ubuntu/src_dir/pyvino
python setup.py install
```

## Test
```
pytest pyvino
```

Test result images are here.
```
$HOME/src_dir/pyvino/pyvino/tests/data
```

# Notebook

```
jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root
``` 
Notebook samples are [HERE](https://github.com/hampen2929/pyvino/tree/master/notebook).

# From the second time
Insted of build and run, use below.
## Start
```
docker start pyvino_gpu1
```

## Execute
```
docker exec -it pyvino_gpu1 bash
```


172.17.0.2

<!-- DISPLAY=172.17.0.1:1
 -->

DISPLAY=192.168.11.8:1

docker network inspect bridge bridge

DISPLAY=172.17.0.2

xhost +
xhost inet:172.17.0.2
xhost local:

import cv2
image = cv2.imread('face.jpg')
cv2.imshow('a', image)

## show

### local
xhost local:

### docker 
xterm
