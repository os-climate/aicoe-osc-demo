
# Demo 2: Preprocessing and Training a NLP model with Open Data Hub 

This demo shows how to use [Open Data Hub](https://opendatahub.io/) (ODH)  tools and infrastructure to define pipelines that preprocess data and train a Natural Language Processing (NLP) model. Specifically, we adapt the data and training pipelines developed by the ALLIANZ NLP team for the OS climate project in this [repository](https://github.com/os-climate/corporate_data_pipeline). The data pipeline takes raw pdf files and extracts text and tables from them, and the training pipeline trains a language model that answers questions based on the extracted text and tables. 
The key components of the ODH infrastructure used in this demo are [JupyterHub](https://JupyterHub-odh-JupyterHub.apps.odh-cl1.apps.os-climate.org/) with a container image, Elyra pipelines with the [kubeflow](http://ml-pipeline-ui-kubeflow.apps.odh-cl1.apps.os-climate.org/) runtime, and a seldon deployment with a service endpoint. The source data, processed data, trained model, and the output data are stored on a bucket on the Ceph S3 storage. The following flowchart depicts the overview of different stages of the project.

![Demo 2 Flowchart](../../docs/assets/demo2-viz.png)

## JupyterHub Image Setup (AICoE-CI, Thoth)

To create a reproducible and shareable notebook development environment for the project, we will build a JupyterHub image with all the dependencies and source code baked in. In this section, we will show how to do this using the ODH toolkit. You can also find a more general and detailed README, along with a walkthrough video tutorial [here](https://www.operate-first.cloud/users/data-science-workflows/docs/create_and_deploy_jh_image.md).

To begin with, you can specify the python dependencies of the project via `Pipfile` / `Pipfile.lock`. Next, you need to set up [`aicoe-ci`](https://github.com/AICoE/aicoe-ci) on your github repo. To do so, follow the steps outlined [here](https://github.com/AICoE/aicoe-ci/tree/21e866d165071978bb857350196819ba74234e3e#setting-aicoe-ci-on-github-organizationrepository). In the `aicoe-ci.yaml` file, add `thoth-build` under the `check` section to enable image builds of your repo. You can configure how the image gets built and where it gets pushed by setting the build parameters described [here](https://github.com/AICoE/aicoe-ci/tree/21e866d165071978bb857350196819ba74234e3e#configuring-build-requirements). For example, we'll set `build-strategy: Dockerfile` and `dockerfile-path: Dockerfile` since we want to build our project image using the `Dockerfile` present at the root of the project. Next, in order to enable AICoE-CI to push the image to the image repository, you can create a robot account and provide it with appropriate permissions as described [here](https://www.operate-first.cloud/users/data-science-workflows/docs/create_and_deploy_jh_image.md#setting-up-the-image-repository) (one-time setup). Finally, once AICoE-CI is set up, you can trigger a build of the image by creating a new tag release on your repo.

You can also use AICoE-CI to enable other Thoth services such as pre-commit checks, CI tests, etc. by adding `thoth-precommit`, `thoth-pytest`, and so on to the `check` section in `.aicoe-ci.yaml`.


## Access JupyterHub Environment

* In order to access the environment for the development of the project, you will have to be added to the odh-env-users group [here](https://github.com/orgs/os-climate/teams/odh-env-users). This can be done by opening an issue on [this page](https://github.com/os-climate/aicoe-osc-demo/issues) with the title "Add <USERNAME> to odh-env-users".

* Once added to the user’s list, use the link to access [JupyterHub](https://JupyterHub-odh-JupyterHub.apps.odh-cl1.apps.os-climate.org).

* After logging into JupyterHub, select the `AICoE OS-Climate Demo` image to get started with the project.

![Spawn JupyterHub](../../docs/assets/demo1-spawn-jupyter.png)


## Data Preprocessing 

Now let’s look at how we process raw data and prepare it for model training. 
The source code for preprocessing is available in the `src` directory preinstalled in the JupyterHub image. This directory follows the project structure laid out in the [aicoe-aiops project template](https://github.com/aicoe-aiops/project-template).

* Extraction

    * In the text extraction notebook, we use pdf2image and pdfminer to extract text from the pdfs and then return json of the extracted text. We then upload the json file to the s3 bucket, and use it later for curation.

    * In the table extraction notebook, we use a pre-trained table detection neural network to extract the image coordinates of all the tables in the input pdfs. Then, we use image processing tools to convert the images of tables into csv files. Finally, we upload these csv files to the s3 bucket, and use it later for curation.

* Curation

    * In the text and table curation notebook, we will load the json files (one per pdf) and the corresponding csv files from the s3 bucket, and then add labels to it. For each text extract or table, we will assign label "1" to the correct corresponding text, and label "0" to a randomly selected text that does not correspond to the table. 


## Training

* Coming soon


## Elyra pipeline

* To access and run the automated Elyra pipeline use this [link](placeholder)

* To setup the Elyra pipeline, use instructions from demo 1 Readme section on [ML pipelines](https://github.com/oindrillac/aicoe-osc-demo/tree/demo1-doc/notebooks/demo1#ml-pipeline)


## Inference Service using Seldon

* Coming soon
