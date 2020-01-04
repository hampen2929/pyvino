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
docker build -t pyvino_image .
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
docker start pyvino
```

## Execute
```
docker exec -it pyvino bash
```
