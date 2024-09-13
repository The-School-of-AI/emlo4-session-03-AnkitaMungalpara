# EMLO4 - Session 03

Docker Compose for MNIST Training, Evaluation, and Inference

In this assignment, you will create a Docker Compose configuration to perform training, evaluation, and inference on the MNIST dataset.

Requirements:

1. You’ll need to use this model and training technique (MNIST Hogwild): https://github.com/pytorch/examples/tree/main/mnist_hogwild
2. Set Num Processes to 2 for MNIST HogWild
3. Create three services in the Docker Compose file: **`train`**, **`evaluate`**, and **`infer`**.
4. Use a shared volume called **`mnist`** for sharing data between the services.
5. The **`train`** service should:
    - Look for a checkpoint file in the volume. If found, resume training from that checkpoint. Train for **ONLY 1 epoch** and save the final checkpoint. Once done, exit.
6. The **`evaluate`** service should:
    - Look for the final checkpoint file in the volume. Evaluate the model using the checkpoint and save the evaluation metrics in a json file. Once done, exit.
    - Share the model code by importing the model instead of copy-pasting it in eval.py
7. The **`infer`** service should:
    - Run inference on any 5 random MNIST images and save the results (images with file name as predicted number) in the **`results`** folder in the volume. Then exit.
8. After running all the services, ensure that the model, and results are available in the **`mnist`** volume.

Detailed Instructions:

1. Build all the Docker images using **`docker compose build`**.
2. Run the Docker Compose services using **`docker compose run train`**, **`docker compose run evaluate`**, and **`docker compose run infer`**. Verify that all services have completed successfully.
3. Check if the checkpoint file (**`mnist_cnn.pt`**) is saved in the **`mnist`** volume. If found, display "Checkpoint file found." If not found, display "Checkpoint file not found!" and exit with an error.
4. Check if the evaluation results file (**`eval_results.json`**) is saved in the **`mnist`** volume.
    1. Example: `{"Test loss": 0.0890245330810547, "Accuracy": 97.12}`
5. Check the contents of the **`results`** folder in the **`mnist`** volume see if the inference results are saved.

The provided grading script will run the Docker Compose configuration, check for the required files, display the results, and perform size and version checks.

You can run it yourself before pushing the code to your repo
