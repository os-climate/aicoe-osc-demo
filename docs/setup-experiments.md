# Set Up Experiments

### Configuring all components of the demo: settings.yaml
[settings.yaml](../notebooks/demo2/settings.yaml) is the central place for tweaking the file paths, extraction, curation, relevance, and kpi-extraction processes. These settings are used by both individual  notebooks and the pipelines. Next, we will go over some important key value pairs in the settings file.

#### config
The config section has the `experiment_name` and `sample_name` that are used as prefix for storing model input and output files. Set them to distinguish your experiments on the bucket. The  `inference_model_path` is the path where the trained relevance and kpi-extraction models are saved. This path will be used by the inference notebook and the pipeline to fetch the trained models and generate inferences.

#### model parameters
Other sections  in the settings correspond to different components of the demo, extraction, curation, relevance and kpi-extraction training and inference. You can tweak the parameters as you see fit and record the results. The default settings are available in the settings file.

### User flow
Before you start running the data processing, training, or inference notebooks, you might want to create a dedicated space (a prefix on the s3 bucket) within which your dataset, models, and results will be stored. This way, our shared s3 bucket will be less cluttered and more organized. You should also select the input pdfs that will be used for inference and training.

#### Default ESG pdfs
To run the demo with default esg pdfs, you can run the setup experiments default pdfs [notebook](../notebooks/demo2/setup_experiments_default_pdfs.ipynb). The most important note here is to set the `sample_name` in the `settings.yaml` file as either `small`, `medium`, or `large`. Based on your selection, the experiments will run with 1, 10, or 145 pdfs respectively. Once that is set, you can run the rest of this notebook to copy the appropriate pdfs, annotations, models, etc. to the prefix created as per the `experiment_name` variable on the s3 bucket.

#### Selectable pdfs
Default pdfs do not offer control over the pdfs that are used for experiments. If you want specfic pdfs and annotation files for your experiments, use the select pdfs [notebook](../notebooks/demo2/setup_experiments_select_pdfs.ipynb). Here you can select pdfs from a defined pool and use them for inference and training.

#### User defined pdfs
Additionally, you can configure where the raw pdfs are situated on the bucket. To follow the conventions of the demo, create a directory and set it's name as the same as `<experiment_name>/<sample_name>`, then add all the pdfs in the directory `<experiment_name>/<sample_name>/pdfs`. If you want to run training, place the annotation files in  `<experiment_name>/<sample_name>/annotations`, and finally place a file like in [kpi-mapping](https://github.com/os-climate/aicoe-osc-demo/tree/master/data/kpi_mapping) in `<experiment_name>/kpi-mapping`

Furthermore, if you want to hack into the settings at a more lower level, and tweak annotations, extraction, inference results, etc., you can update the parameters defined in the [config.py](../notebooks/demo2/config.py) file. However, it is not recommended.

Note: if you don't have access to the s3 bucket, please open an issue [here](https://github.com/os-climate/OS-Climate-Community-Hub/issues/new?assignees=erikerlandson&labels=Bucket-credentials&template=request-credentials-for-an-os-climate-bucket.md&title=Request+credentials+for+an+os-climate+bucket).
