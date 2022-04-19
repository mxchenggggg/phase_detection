# Surgical Phase Detection Using Deep Learning

## About the project
This is a course project for CIS II Spring 2022.

## Environment
The docker image for this project is based on `nvidia/cuda:11.3.1-devel-ubuntu18.04`. Refer to `dockerfiles/pytorch.txt` for PyTorch version and `dockerfiles/requirements.txt` for other dependencies.

`docker.sh` provides basic CML options to manage docker image and container. See usage details with `./docker.sh -h`

1. run container: `./docker.sh -r`
2. new terminal in running contrainer: `./docker.sh -n
3. delete container: `./docker.sh -d`

## Team Workflow

### Remote Server
1. All dependencies are managed with docker. Do not add anything to the server.
2. ./docker.sh -b builds the docker image, it will delete current container first, make sure no one is using the container before doing that.
3. Use docker ps to list running container, if the container is already running, use ./docker.sh -n to open a new terminal in the running container. Running ./docker.sh -r in this case will attach you to other's terminal.
4. Open a tmux session, then access docker container (either ./docker.sh -r or -n).

### Development Environment
To use full functionality of VSCode, developing in docker container is recommended. Follow steps below to connect to a remote container:
1. start a container in remote server
2. use Remote-SSH extension to connect to remote server (a new VSCode window pops up)
3. open a folder in the remote server (in the VSCode widow from previous step)
4. use Remote-Container extension to connect to the running container in remote server 

### Result Evaluation
For each model save per video results in a single pickle file with convention [(video_index,pred,ground_truth)...]
1. Run bars_cms.py to draw bar plots, generate CMs and model summaries for all three models. 
2. Run bar_cm.py to draw bar plot, generate CM and model summary fot one single model.
3. Need to modify model name and pickle file name in these files if not using conventional names: "SVRC, TeCNO, TRANS".
