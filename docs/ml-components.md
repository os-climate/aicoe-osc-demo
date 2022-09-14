# Model Components

## Data Preprocessing

Now let’s look at how we process raw data and prepare it for model training.
The source code for preprocessing is available in the `src` directory preinstalled in the JupyterHub image. This directory follows the project structure laid out in the [aicoe-aiops project template](https://github.com/aicoe-aiops/project-template).

* Extraction

    * In the text extraction notebook, we use pdf2image and pdfminer to extract text from the pdfs and then return json of the extracted text. We then upload the json file to the s3 bucket, and use it later for curation.

    * In the table extraction notebook, we use a pre-trained table detection neural network to extract the image coordinates of all the tables in the input pdfs. Then, we use image processing tools to convert the images of tables into csv files. Finally, we upload these csv files to the s3 bucket, and use it later for curation.

* Curation

    * In the text and table curation notebook, we will load the json files (one per pdf) and the corresponding csv files from the s3 bucket, and then add labels to it. For each text extract or table, we will assign label "1" to the correct corresponding text, and label "0" to a randomly selected text that does not correspond to the table.

## Training

* Train Relevance
    * In the train relevance notebook, we use the extracted and curated text from the data processing steps to train a relevance classifier model. That is, a model that takes a text paragraph as input, and determines whether it is relevant for answering a given KPI question. We also calculate some performance metrics for this trained model, namely the accuracy, precision, recall, and F1 scores on each of the KPIs. Finally, we save the model as well as the performance metrics on a s3 bucket for later use.

* Train KPI Extraction
    * Once we have a trained relevance classifier, we run it on the PDFs to get relevant paragraphs for each KPI. Then, in the train KPI extraction notebook, we re-format the relevance results into the SQuAD dataset format. Next, we proceed to train a question-answering (QA) model on this SQuAD formatted dataset. Then, we evaluate this QA model in terms of F1 scores on each of the KPIs. Finally, we save the trained model as well as the performance metrics on a s3 bucket.

## Inference

* Infer relevance
    * The infer relevance notebook takes in extracted text from the preprocessing stage and for a predefined set of KPI questions, finds relevant paragraphs from the text. These paragraphs are then used to find the exact answers to the questions. The notebook uses a fine-tuned language model stored on s3. The output prediction csv files are saved back on s3.

* Infer KPI
    * The infer kpi notebook takes in the results from the infer relevance stage and for the predefined set of KPI questions, it finds the exact answers from the relevant paragraphs. The notebook uses a fine-tuned language model stored on s3. The output prediction csv files are saved back on s3.

## Trino

* Results table
    * The create results table notebook takes the prediction output csv from infer KPI step and creates a Trino SQL table that can be used for querying and visualization in Superset.
