# Usage
__Todo: update this document__

- build docker container
- run `train.py`
```
docker run --gpus '"device=0"' -it --shm-size=20gb --rm -v /home/resnant/work/materials-concept-learning/src:/workspace exp_container:latest python /workspace/train.py -p example_dml.json
```
