# Dockerfiles

Here you can find the Dockerfiles used to generate the different Docker images available at [dockerhub](https://hub.docker.com/repository/docker/sensioai/nangs).

- [gpu](./gpu): Base image with CUDA, Pytorch and nangs installed for GPU.
- [gpu-jupyter](./gpu-jupyter): Install jupyter on top of [gpu](./gpu) to work with notebooks.
- [cpu](./cpu): Base image with Pytorch and nangs installed for CPU.
- [cpu-jupyter](./cpu-jupyter): Install jupyter on top of [cpu](./cpu) to work with notebooks.
- [dev](./dev): Image for developing nangs. It includes CUDA, Pytorch and [nbdev](http://nbdev.fast.ai/), the framework used for developing the library with notebooks (automatic code generation, testing, documentation ...).

The *latest* and *jupyter* tags use the *gpu* images.