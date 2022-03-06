#!/usr/bin/env bash
username=`whoami`
docker_image_name="surgical_phase_detection:base"
docker_container_name="surgical_phase_detection_${username}"

function docker_del () {
    running_count=`docker ps | grep -w ${docker_container_name} | wc -l`
    total_count=`docker ps -a | grep -w ${docker_container_name} | wc -l`
    if [ $running_count -eq 1 ];then
        docker stop ${docker_container_name}
        echo "stopped ${docker_container_name}"
    fi
    if [ $total_count -eq 1 ];then
        docker rm ${docker_container_name}
        echo "deleted ${docker_container_name}"
    fi
}

function docker_run(){
    count=`docker ps -a | grep -w ${docker_container_name} | wc -l`
    if [ $count -eq 0 ];then
        echo "docker: ${docker_container_name} does not exist, start one"
        docker run -it \
	        --gpus all \
            --device /dev/dri \
		    --name ${docker_container_name} \
            --network host \
            --ipc host\
            -v ${PWD}:/home/ubuntu/phase_detection \
            -v /home/wluo14/data:/home/ubuntu/data \
            -w /home/ubuntu/phase_detection  ${docker_image_name} bash
    elif [ $count -eq 1 ];then
        echo "docker: ${docker_container_name} exists, restart it"
	    docker start ${docker_container_name}
        docker attach ${docker_container_name}
    else
        docker ps -a | grep -w ${docker_container_name}
        echo "please delete redundant docker container"
    fi
}

function docker_build () {
    docker_del
    DOCKER_BUILDKIT=1 docker build -t ${docker_image_name}\
        --build-arg USER_ID=$(id -u) \
        --build-arg GROUP_ID=$(id -g) \
        ./dockerfiles
    docker_run
}

function docker_new_ternimal () {
    count=`docker ps | grep -w ${docker_container_name} | wc -l`
    if [ $count -eq 1 ];then
        docker exec -it ${docker_container_name} /bin/bash
    elif [ $num -eq 0 ];then
        echo "${docker_container_name} does not exist"
    else
        docker ps -a | grep -w ${docker_container_name}
        echo "please delete redundant docker"
    fi
}

function print_help() {
    echo "
    -h | help info
    -b | build image and run a container
    -r | run the container, creat one if none exits
    -n | open a new terminal attached to running container
    -d | delete the running container
    "
}

while getopts "rdnbh" arg
do
    case $arg in
        r)
            docker_run
            exit 0
            ;;
        d)
            docker_del
            exit 0
            ;;
        n)
            docker_new_ternimal
            exit 0
            ;;
        b)
            docker_build
            exit 0
            ;;
        h)
            print_help
            exit 0
            ;;    
    esac
done