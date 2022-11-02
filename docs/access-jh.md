# Access JupyterHub Environment

* In order to access the environment for the development of the project, you will have to be added to the odh-env-users group [here](https://github.com/orgs/os-climate/teams/odh-env-users). This can be done by opening an issue on [this page](https://github.com/os-climate/OS-Climate-Community-Hub/issues/new) with the title "Add <USERNAME> to odh-env-users".

* Once added to the user’s list, you should be able to access [JupyterHub](https://jupyterhub-odh-jupyterhub.apps.odh-cl2.apps.os-climate.org), [Kubeflow Pipelines](http://ml-pipeline-ui-kubeflow.apps.odh-cl2.apps.os-climate.org/), [Trino](https://cloudbeaver-odh-trino.apps.odh-cl1.apps.os-climate.org/), and [Superset Dashboard](https://superset-secure-odh-superset.apps.odh-cl1.apps.os-climate.org/). Go ahead and log into the JupyterHub instance.

* After logging into JupyterHub, select the `AICoE OS-Climate Demo` image to get started with the project.

![Spawn JupyterHub](assets/demo2-spawn-jupyter.png)

* Make sure you add the credentials file to the root folder of the project repository. For more details on how to set up your credentials, and retrieve your JWT token to access Trino, refer to the documentation given [here](https://github.com/os-climate/os_c_data_commons/blob/main/docs/setup-initial-environment.md#4-set-your-credentials-environment-variables).

* To install the dependencies needed for running the notebooks, you can run a `pipenv install` at the root of the repository or use the [Horus](https://github.com/thoth-station/jupyterlab-requirements/blob/dc92a4b14f539e6f464b3f202355242b4f729e13/docs/source/horus-magic-commands.md) magic commands from within the notebook.
