# Surgical Phase Detection Using Deep Learning

## About the project
This is a course project for CIS II Spring 2022.

## Environment
The docker image for this project is based on `nvidia/cuda:11.3.1-devel-ubuntu18.04`. Refer to `dockerfiles/pytorch.txt` for PyTorch version and `dockerfiles/requirements.txt` for other dependencies.

`docker.sh` provides basic CML options to manage docker image and container. See usage details with `./docker.sh -h`

1. run container: `./docker.sh -r`
2. new terminal in running contrainer: `./docker.sh -n
3. delete container: `./docker.sh -d`