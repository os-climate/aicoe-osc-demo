# Elyra pipeline

To run the notebooks in a sequential and automated fashion, we use the Elyra notebook pipelines editor and kubeflow pipelines to ensure the workflow is automated.

You can access the saved Elyra Pipeline for Data Preprocessing (extraction + curation) [here](../notebooks/demo2/preprocessing.pipeline), and the one for end-to-end Inference (extraction + infer relevance + infer kpi + create tables) [here](../notebooks/demo2/inference.pipeline)

In order to set up an Elyra pipeline and run it on kubeflow pipelines, you can follow the instructions below:

### Setup Kubeflow Pipelines

* To run the pipeline, you will need the Elyra notebook pipeline editor. Make sure you are either on the `Custom Elyra Notebook Image` or the `AICoE OS-Climate Demo` image on the OS-Climate [JupyterHub](https://jupyterhub-odh-jupyterhub.apps.odh-cl2.apps.os-climate.org/)

* To get a Kubeflow pipeline running, you need to create a runtime image and a kubeflow pipelines runtime configuration.

#### Add runtime images

To create a runtime image using Elyra, follow the steps given [here](https://github.com/AICoE/elyra-aidevsecops-tutorial/blob/master/docs/source/create-ai-pipeline.md#add-runtime-images-using-ui).

![Add runtime images](assets/demo2-runtime-image.png)

Fill all required fields to create a runtime image for the project repo:

- Name: `aicoe-osc-demo`
- Image Name: `quay.io/os-climate/aicoe-osc-demo:v0.17.0`

#### Add Kubeflow Pipeline runtime configuration

To create a kubeflow pipeline runtime configuration image using Elyra, follow the steps given [here](https://github.com/AICoE/elyra-aidevsecops-tutorial/blob/master/docs/source/create-ai-pipeline.md#create-runtime-to-be-used-in-kubeflow-pipeline-using-ui).

![Add runtime configuration](assets/demo2-runtime-configuration.png)

Insert all inputs for the Runtime"
- Name: `aicoe-osc-demo-cl2`
- Kubeflow Pipeline API Endpoint: `http://ml-pipeline-ui.kubeflow.svc.cluster.local:80/pipeline`
- Kubeflow Pipeline Engine: `Tekton`
- Authentication Type: `NO_AUTHENTICATION`
- Cloud Object Storage Endpoint: `S3_ENDPOINT`
- Cloud Object Storage Username: `AWS_ACCESS_KEY_ID`
- Cloud Object Storage Password: `AWS_SECRET_ACCESS_KEY`
- Cloud Object Storage Bucket Name: `S3_BUCKET`
- Cloud Object Storage Credentials Secret: `S3_SECRET`

#### Set up Notebook Properties

Now you can create Elyra pipelines by clicking on the "+" button in your JupyterHub environment, selecting "Kubeflow Pipeline Editor" under Elyra, and then clicking and dragging notebooks to the editor window. Then, you can connect the notebooks to create your workflow. This is what the inference pipeline in this demo looks like, after this step:

![Elyra Pipeline](assets/demo2-elyra-pipeline.png)

To trigger this pipeline, you need to make sure that the node properties for each notebook have been updated.

![Set up Notebook Properties](assets/demo2-notebook-properties.png)

You need to set the cloud object storage details like `S3_ENDPOINT`, `S3_BUCKET` as well as the Trino database access credentials like `TRINO_USER`, `TRINO_PASSWD`, `TRINO_HOST`, `TRINO_PORT` as environment variables. Please note, if you are using the Cloud Object Storage Credentials Secret field in the Kubeflow Pipelines Runtime configuration, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` can be omitted from the notebook properties as kubeflow is automatically able to read the encrypted credentials from the secret defined in OpenShift. If not, then you will have to add `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` as environment variables here.

#### Run Pipeline

Once your pipeline is set up, you can run the pipeline by hitting the run button on the top left of the pipeline. You can give it a name and select the previously created Kubeflow Pipelines Runtime Configuration from the dropdown.

![Run Pipeline](assets/demo2-run-pipeline.png)


## Superset Visualization

* The Superset dashboard is the final step of demo 2. The automated Elyra inference pipeline answers KPI questions from raw pdfs and stores the results in the Trino table. The dashboard queries the table according to user selected filters and shows the answers. To interact with the results, find the [dashboard here](https://superset-secure-odh-superset.apps.odh-cl1.apps.os-climate.org/superset/dashboard/15).
