# OpenCV Base Image

Dockerfile to build the OpenCV base image. Can be built on x86 CPU or Nvidia GPU.

## Building on CPU

```
docker build -t opencv .
```
## Building on GPU

Ensure that you are running on the Jeston NX device and have Nvidia as the default runtime.

Update `/etc/docker/daemon.json` to the following:

```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```

Restart the docker service

```
systemctl restart docker
```

Build the image

```
docker build -f Dockerfile.cuda -t opencv .
```
