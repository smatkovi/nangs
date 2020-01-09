# NANGS

A Python module built on top of Pytorch to solve Partial Differential Equations. 

Our objective is to develop a new tool for simulating nature, using Neural Networks as solution approximation to Partial Differential Equations, increasing accuracy and optimziation speed while reducing computational cost.

Read our [paper](https://arxiv.org/abs/1912.04737) to know more.

## Examples

You can learn *nangs* with our examples, where we solve different PDEs in different use cases.

- First steps [tutorial](./ipynb/tutorial.ipynb).
- One-dimensional Advection Equation [example](./ipynb/adv1d/adv1d.ipynb).
- Two-dimensional Advection Equation [example](./ipynb/adv2d/adv2d.ipynb).
- Smith-Hutton [problem](./ipynb/smith-hutton/smithHutton.ipynb).
- Smith-Hutton [problem](./ipynb/smith-hutton/smithHutton2.ipynb) including boundary conditions in the solution.

## Instructions

To run the examples, first clone this repository

```
git clone https://github.com/juansensio/nangs.git
cd nangs
```

Then build the docker image 

```
# build CPU version
docker build -t nangs docker/cpu

# build GPU version
docker build -t nangs docker/gpu
```

Finally, run the container
```
./run.sh <your-notebook-token>
```

You should now be able to run the notebooks on [localhost:8888](http://localhost:8888). If you use the CPU version and did not install nvidia-docker, remove the *gpus* option from the *run.sh* script.

## Dependencies

- Docker 19.03

If you have an NVIDIA GPU:

- NVIDIA driver (you do not need to install the CUDA toolkit)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 
