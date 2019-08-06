# USAGE - ./launch-docker-cpu.sh {abs-path-to-GDL-code}
docker run --rm --network=host -it -v $1:/GDL gdl-image-cpu
