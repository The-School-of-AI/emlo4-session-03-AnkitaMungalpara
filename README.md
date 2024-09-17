# MNIST Training, Evaluation, and Inference with Docker Compose

This project provides a Docker Compose configuration to handle training, evaluation, and inference on the [MNIST Hogwild](https://github.com/pytorch/examples/tree/main/mnist_hogwild) dataset with PyTorch. It uses Docker Compose to orchestrate three services: **train**, **evaluate**, and **infer**.

## Table of Contents

- [Requirements](#requirements)
- [Docker Compose Services](#docker-compose-services)
   - [Train](#1-train)
   - [Evaluate](#2-evaluate)
   - [Infer](#3-infer)
- [Command-Line Arguments](#command-line-arguments)
- [Docker Compose Configuration](#docker-compose-configuration)
- [Instructions](#instructions)


## Requirements

- `torch`
- `torchvision`

You can instll requirements using below command:
```bash
pip install -r requirements.txt
```

## Docker Compose Services

The Docker Compose configuration defines three services:

### 1. `train`

- Trains the MNIST model.
- Checks for a checkpoint file in the shared volume. If found, resumes training from that checkpoint.
- Saves the final checkpoint as `mnist_cnn.pt` and exits.

### 2. `evaluate`

- Checks for the final checkpoint (`mnist_cnn.pt`) in the shared volume.
- Evaluates the model and saves metrics in `eval_results.json`.
- The model code is imported rather than copy-pasted into `eval.py`.

### 3. `infer`

- Runs inference on 5 random MNIST images.
- Saves the results (images with predicted numbers) in the `results` folder within the docker container and exits.


## Command-Line Arguments

The MNIST training script accepts the following command-line arguments:

| Argument         | Description                                                        | Default   |
|------------------|--------------------------------------------------------------------|-----------|
| `--batch-size`   | Input batch size for training                             | 64        |
| `--epochs`       | Number of epochs to train                                     | 1         |
| `--lr`           | Learning rate                                                 | 0.01      |
| `--momentum`     | SGD momentum                                                   | 0.5       |
| `--seed`         | Random seed                                                   | 1         |
| `--log-interval` | How many batches to wait before logging training status            | 10        |
| `--num-processes`| Number of processes to run script on for distributed processing                             | 2         |
| `--dry-run`      | Quickly check a single pass without full training                | False     |
| `--save_model`   | Flag to save the trained model                               | True      |
| `--save-dir`     | Directory where the checkpoint will be saved                 | `./`      |


This table provides a clear and concise overview of the available command-line arguments and their default values.


## Docker Compose Configuration

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  train:
    # Train service
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - mnist:/opt/mount
      - ./model:/opt/mount/model
      - ./data:/opt/mount/data

  evaluate:
    # Evaluate service
    build:
      context: .
      dockerfile: Dockerfile.eval
    volumes:
      - mnist:/opt/mount
      - ./model:/opt/mount/model
      - ./data:/opt/mount/data

  infer:
    # Inference service
    build:
      context: .
      dockerfile: Dockerfile.infer
    volumes:
      - mnist:/opt/mount
      - ./data:/opt/mount/data

volumes:
  mnist:
```

## Instructions

1. **Build Docker Images**:
   ```bash
   docker compose build
   ```

2. **Run Services**:
   - Train:
     ```bash
     docker compose run train
     ```
   - Evaluate:
     ```bash
     docker compose run evaluate
     ```
   - Inference:
     ```bash
     docker compose run infer
     ```

3. **Verify Results**:
   - **Checkpoint File**: Check if `mnist_cnn.pt` is in the `mnist` volume.
     - If found: "Checkpoint file found."
     - If not found: "Checkpoint file not found!" and exit with an error.
   - **Evaluation Results**: Verify `eval_results.json` in the `mnist` volume.
     - Example format: `{"Test loss": 0.0890245330810547, "Accuracy": 97.12}`
   - **Inference Results**: Check the `results` folder in the `mnist` volume for saved images.

