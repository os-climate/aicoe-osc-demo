# How to contribute

aicoe-osc-demo welcomes contributions from developers and users. Follow the instructions in this section for a smooth interaction. The easiest way to contribute is to create [issues](https://github.com/os-climate/aicoe-osc-demo/issues) that you encounter while running the demo.  Someone from the team will review the issue and  provide support with the demo.

### Creating a Pull Request (PR)
The best way to contribute to the demo is to contribute code. Follow the instructions [here](https://github.com/aicoe-aiops/data-science-workflows/blob/master/docs/develop_collaborate/how-to-contribute.md#contributing-via-pull-requests) to learn how to create a PR.

The pull request will be checked for 3 automatic tests:

#### Pre-commit
Pre-commit ensures that the code is free of any style errors. Use the following commands after `git add <file>`

`pip install pre-commit`

`pre-commit -run --all-files`

Read and fix the changes suggested by  pre-commit and  then proceed to  add the files and commit.

#### Developer Certificate of Origin (DCO)
DCO is a way to verify the identity of the developer. To pass this test, you will have to sign  your commit message.

`git commit -m  "_commit-message_ Signed-off-by _name_ <_email_id_>"

Note: Creating a PR with one commit and one logical change makes reviewing the code much easier and increasing the chances of getting it merged faster. You can use the squash commit feature to collapse multiple commits into single commit.

#### Image build
This test ensures that the container image for this repository works, If you plan to make changes to the build pipeline components such as the docker or pip files, make sure that this test passes. In all other cases, it should automatically pass.

### Reviewing a PR
Another way of contributing to the project is to review an active PR.  Follow the [best practices](https://github.com/aicoe-aiops/data-science-workflows/blob/master/docs/develop_collaborate/how-to-contribute.md#reviewing-pull-requests) for reviewing a PR.
