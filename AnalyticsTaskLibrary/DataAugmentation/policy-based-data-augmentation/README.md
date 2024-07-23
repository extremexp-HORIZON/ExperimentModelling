# Policy-based Data Augmentattion

## Introduction
This repository contains the code for a policy-based data augmentation task. The goal of this project 
is to provide a flexible and efficient way to augment datasets using policy-based approaches, enhancing 
the performance of machine learning models.

## Setting up the Docker with the API
To facilitate the deployment and usage of the data augmentation API, we provide a Docker setup. 
Follow these steps to set up the Docker environment:

1. Clone the Repository
```bash
git clone repo.git
cd policy-based-data-augmentation
```

2. Build the Docker image
```bash
docker build -t policy-based-data-augmentation .
```

3. Run the Docker container
```bash
docker run -p 5000:5000 policy-based-data-augmentation
```
This will expose the API on port 5000. Open your browser and navigate to http://localhost:5000 to 
check if the API is running successfully.


## API Documentation
The API provides a different endpoints to perform data augmentation. For detailed information 
on the available API endpoints and how to interact with the data augmentation service an 
API documentation is provided. To access the API documentation, go to http://localhost:5000/docs
after running the Docker container.

## Hands-On 
To demonstrate the usage of the code directly from the source, we have included a Python notebook 
(example.ipynb). Follow these steps to run the notebook:

1. Install the required packages
```bash
pip install -r requirements.txt
```
or alternatively, use the provided `poetry` environment:
```bash
poetry install
```

2. Start the Jupyter notebook
```bash
jupyter notebook
```
This will open a new tab in your browser. Navigate to the `/hands-on` examples and run the cells. 



