# Usage
Please follow the description below to train the deep metric learning (materials concepts learning) model.

## Building Docker Container
To build the Docker container, use the following command:
```
docker build -t mat_concept_learn:v1.0 ./docker
```

## Running the training script
Run the following command to execute the script:

```
docker run --gpus '"device=0"' -it --shm-size 20gb --rm -v "$(pwd)":/workspace mat_concept_learn:v1.0 python /workspace/train.py -p example_dml.json
```

- Make sure to replace `example_dml.json` with the name of your parameter file.
- The dataset will be automatically downloaded in the first run.
- You can modify the --gpus flag to specify which GPU(s) to use for training.
- Note: This code assumes that you have already installed Docker on your system.