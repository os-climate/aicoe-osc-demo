AI CoE Demo Repo for OS-Climate
==============================

This repository is the central location for the demos the AI CoE team is developing within the [OS-Climate](https://github.com/os-climate) project.

## Demo 1 - ETL & Dashboarding

This demo provides notebooks and an Elyra pipeline that provide an example of how to use the tools available with [Open Data Hub](https://opendatahub.io/) on an [Operate First](https://www.operate-first.cloud/) cluster to perform ETL and create interactive dashboards and visualizations of our data.

* Ingest raw data into Trino ([Notebook](https://github.com/os-climate/aicoe-osc-demo/blob/master/notebooks/demo1-create-tables.ipynb))
* Run a join against a different federated data table ([Notebook](https://github.com/os-climate/aicoe-osc-demo/blob/master/notebooks/demo1-join-tables.ipynb))
* Collect the results ([Pipeline](https://github.com/os-climate/aicoe-osc-demo/blob/master/notebooks/demo1.pipeline))
* Visualize it in Superset ([Dashboard](https://superset-secure-odh-superset.apps.odh-cl1.apps.os-climate.org/superset/dashboard/3/))


## Demo 2 - Model Training and Serving

This demo provides notebooks and an Elyra pipeline that provide an example of how to use the tools available with [Open Data Hub](https://opendatahub.io/) on an [Operate First](https://www.operate-first.cloud/) cluster to train an NLP model and deploy it as a service.

* Train an NLP model
* Track performance metrics
* Store model in remote storage
* Deploy model inference service
