# USAGE - ./launch-docker-gpu.sh {abs-path-to-GDL-code}
docker run --rm --runtime=nvidia -p 8888:8888 -it -v $1:/GDL gdl-image-gpu

