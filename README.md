# teched25-sapaicore-byom
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/itsmarlo/teched25-sapaicore-byom)

This repository contains the source code for the SAP TechEd 2025 demonstration on implementing a "Bring-Your-Own-Model" (BYOM) workflow on SAP AI Core. It provides a complete, end-to-end example of training a text classification model to identify reasons for blocked invoices and deploying it as a scalable inference service.

The project is structured for production-readiness, showcasing best practices such as Docker containerization, pipeline-based orchestration, and API-driven MLOps.

## Project Goal

The primary goal of this project is to build an AI model that automatically classifies the reason an invoice is blocked based on its descriptive text. This helps automate a common accounts payable process, reducing manual effort and speeding up resolution times. The classification categories include:

- Price Variance
- Quantity Mismatch
- Missing PO Reference
- Three-Way Match Failure
- Supplier Issues

## End-to-End Workflow

The entire Machine Learning workflow is orchestrated by SAP AI Core and can be broken down into the following stages:

1.  **Data Preparation**: A synthetic dataset mimicking blocked invoice descriptions from an SAP S/4HANA system is generated using `data_preparation.py`.
2.  **Training Pipeline**: A containerized training job is executed on SAP AI Core. It fetches the dataset from an S3 object store, fine-tunes a `microsoft/deberta-v3-base` model from Hugging Face, evaluates its performance, and registers the final trained model as an artifact.
3.  **Serving Pipeline**: Another containerized job takes the trained model artifact and deploys it as a RESTful API using KServe. The service is configured for auto-scaling.
4.  **Inference**: The deployed API endpoint can then be called with new invoice texts to get real-time predictions for why an invoice is blocked.

## Repository Structure

```
.
├── AI Core TechEd.postman_collection.json # Postman collection for interacting with SAP AI Core APIs.
├── data_preparation.py                    # Script to generate the synthetic invoice dataset.
├── data/                                  # Contains the generated dataset splits and label map.
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── label_map.json
├── pipelines/                             # SAP AI Core workflow and serving templates.
│   ├── training-template.yaml             # Argo-based workflow for the training pipeline.
│   └── serving-template.yaml              # KServe-based template for the inference service.
├── training/                              # Code and configuration for model training.
│   ├── Dockerfile.train                   # Dockerfile to build the training container.
│   ├── requirements-train.txt             # Python dependencies for training.
│   └── train.py                           # The main training script using Hugging Face Transformers.
└── serving/                               # Code and configuration for model serving.
    ├── Dockerfile.serve                   # Dockerfile to build the serving container.
    ├── requirements-serve.txt             # Python dependencies for the API.
    ├── serve.sh                           # Entrypoint script for the serving container.
    └── serving.py                         # FastAPI application for inference.
```

## Getting Started

### Prerequisites

*   An SAP Business Technology Platform (BTP) account with SAP AI Core configured.
*   An S3-compatible object store (e.g., AWS S3) and its access credentials.
*   Docker installed and running.
*   A container registry (like Docker Hub) to host your images.
*   Python 3.10 or later.
*   Postman for making API calls.

### Step-by-Step Guide

#### 1. Clone the Repository

```bash
git clone https://github.com/itsmarlo/teched25-sapaicore-byom.git
cd teched25-sapaicore-byom
```

#### 2. Prepare and Upload Data

First, generate the synthetic dataset.

```bash
python data_preparation.py
```

This will create `train.csv`, `validation.csv`, `test.csv`, and `label_map.json` inside the `data/` directory.

Next, upload these files to your S3 bucket. For this example, we assume they are placed under a path like `s3://<your-bucket-name>/data/invoices/`. Your S3 structure should look like this:

```
<your-bucket-name>/
└── data/
    └── invoices/
        ├── train.csv
        ├── validation.csv
        ├── test.csv
        └── label_map.json
```

#### 3. Build and Push Docker Images

You need to build the training and serving Docker images and push them to a registry that your SAP AI Core instance can access.

**Important**: Before building, update the image names in `pipelines/training-template.yaml` and `pipelines/serving-template.yaml` from `docker.io/itsmarlo/...` to point to your own container registry (e.g., `docker.io/<your-username>/...`).

```bash
# Set your Docker Hub username
export DOCKER_USER=your-docker-hub-username

# Build and push the training image
docker build -t $DOCKER_USER/invoice-train:latest -f training/Dockerfile.train .
docker push $DOCKER_USER/invoice-train:latest

# Build and push the serving image
docker build -t $DOCKER_USER/invoice-serve:latest -f serving/Dockerfile.serve .
docker push $DOCKER_USER/invoice-serve:latest
```

#### 4. Configure SAP AI Core via Postman

This project uses a Postman collection to interact with SAP AI Core.

1.  **Import Collection**: Import the `AI Core TechEd.postman_collection.json` file into Postman.
2.  **Set Up Environment**: Create a Postman environment and configure the following variables based on your SAP AI Core service key:
    *   `AI_API_URL`: The `url` from your service key.
    *   `tokenURL`: The `uaa.url` from your service key.
    *   `clientId`: The `uaa.clientid` from your service key.
    *   `clientSecret`: The `uaa.clientsecret` from your service key.
3.  **Update Requests**: You will need to modify some of the request bodies in the collection to match your specific setup:
    *   **Set up > Create Object Store Secrets**: Replace the placeholder `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `bucket` with your S3 details.
    *   **Set up > Onboard GitHub repo**: Update the `url` and credentials if you are using a fork of this repository.
    *   **Training > Register dataset artifact**: Update the `url` to point to your S3 path (e.g., `ai://<your-s3-secret-name>/data/invoices/`).
    *   **Training > Create configuration**: After registering the dataset, replace `REPLACE_WITH_DATASET_UUID` with the `id` of the artifact you just created.
    *   **Serving > Create configuration**: After a successful training run, find the model artifact ID from the execution logs or by querying the API. Replace `<MODEL_ARTIFACT_ID>` with this ID.

#### 5. Execute the MLOps Workflow

Run the Postman requests in the following order:

1.  **Authentication**: In the `Training` folder, run `fetch tocken` to get an `access_token`. This token is automatically stored as a collection variable.
2.  **Setup**: Execute the requests under the `Set up` folder to configure your resource group, repository, and secrets.
3.  **Training**:
    *   Run `Register dataset artifact` to make your S3 data known to AI Core.
    *   Run `Create configuration` to define the training job parameters.
    *   Run `Trigger execution` to start the training pipeline. You can monitor its status in the SAP AI Core UI.
4.  **Serving**:
    *   Once training is complete, retrieve the model artifact ID from the `outputs` of the training execution.
    *   Update and run `Create configuration` under the `Serving` folder using the new model artifact ID.
    *   Run `Create deployment` to deploy the model as an API endpoint.
5.  **Inference**:
    *   Find the deployment URL in the SAP AI Core UI or from the deployment details API response.
    *   Use the `Get prediction` request (after updating its URL) to send text and receive a classification.

## Technical Stack

-   **Orchestration**: SAP AI Core
-   **Containerization**: Docker
-   **ML Framework**: PyTorch
-   **NLP Library**: Hugging Face Transformers (`microsoft/deberta-v3-base`)
-   **API Framework**: FastAPI
-   **API Client**: Postman
