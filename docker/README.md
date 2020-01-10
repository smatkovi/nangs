# Dockerfiles

Here you can find the Dockerfiles used to generate the different Docker images available at [dockerhub](https://hub.docker.com/r/sensioai/nangs).

- [gpu](https://github.com/juansensio/nangs/tree/master/docker/gpu): Base image with CUDA, Pytorch and nangs installed for GPU.
- [gpu-jupyter](https://github.com/juansensio/nangs/tree/master/docker/gpu-jupyter): Install jupyter on top of *gpu* to work with notebooks.
- [cpu](https://github.com/juansensio/nangs/tree/master/docker/cpu): Base image with Pytorch and nangs installed for CPU.
- [cpu-jupyter](https://github.com/juansensio/nangs/tree/master/docker/cpu-jupyter): Install jupyter on top of *cpu* to work with notebooks.
- [dev](https://github.com/juansensio/nangs/tree/master/docker/dev): Image for developing nangs. It includes CUDA, Pytorch and [nbdev](http://nbdev.fast.ai/), the framework used for developing the library with notebooks (automatic code generation, testing, documentation...).

The *latest* and *jupyter* tags use the *gpu* images.