# Using Open Data Hub toolkit and Operate First infrastructure for OS-Climate

This repository is the central location for the demos the Open Services ( previously AICoE) team is developing within the [OS-Climate](https://github.com/os-climate) project.

The following demos provide examples of how to use the tools available with [Open Data Hub](https://opendatahub.io/) toolkit on an [Operate First](https://www.operate-first.cloud/) cluster to perform ETL, create inference pipelines, create interactive dashboards and visualizations of our data.

## [Demo 1 - ETL & Dashboarding](notebooks/demo1/README.md)

* [Ingest raw data from S3 as tables on Trino](notebooks/demo1/demo1-create-tables.ipynb)
* [Run SQL queries from a Jupyter Notebook environment](notebooks/demo1/demo1-join-tables.ipynb)
* [Demo 1 Elyra Pipeline](https://github.com/os-climate/aicoe-osc-demo/blob/master/notebooks/demo1/demo1.pipeline)
* [Results visualized on a Superset Dashboard](https://superset-secure-odh-superset.apps.odh-cl1.apps.os-climate.org/superset/dashboard/3/)
* [Video on creating Elyra Pipelines and Superset Dashboard](https://youtu.be/TFgsR7UlcHA)


## [Demo 2 - Automated Inference Pipeline](notebooks/demo2/README.md)

* [Steps to Create a JupyterHub image from the repository](notebooks/demo2/README.md#jupyterhub-image-setup-aicoe-ci-thoth)
* [Automated inference pipeline](https://github.com/os-climate/aicoe-osc-demo/blob/master/notebooks/demo2/inference.pipeline)


## [Demo 3 - ELT and Dashboarding](notebooks/demo3/README.md)

* [Extract, Load and Transform raw KPIs into useful data and metadata](notebooks/demo3/README.md)
* [Results visualized on a Superset Dashboard](https://superset-secure-odh-superset.apps.odh-cl1.apps.os-climate.org/superset/dashboard/15)
* [Video on creating JupyterHub image and inference pipeline](https://youtu.be/lGeT615YNlM)
