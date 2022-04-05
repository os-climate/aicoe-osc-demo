FROM quay.io/thoth-station/s2i-elyra-custom-notebook:v0.4.5
LABEL "name"="aicoe-osc-demo" \
      "io.openshift.s2i.build.image"="quay.io/thoth-station/s2i-elyra-custom-notebook:v0.4.5" \
      "io.openshift.s2i.scripts-url"="image:///opt/app-root/builder"

ENV JUPYTER_ENABLE_LAB="1" \
    ENABLE_MICROPIPENV="1" \
    THAMOS_RUNTIME_ENVIRONMENT="" \
    THOTH_ADVISE="0" \
    THOTH_ERROR_FALLBACK="1" \
    THOTH_DRY_RUN="1" \
    THAMOS_DEBUG="0" \
    THAMOS_VERBOSE="1" \
    THOTH_PROVENANCE_CHECK="0" \
    GIT_SSL_NO_VERIFY=true \
    GIT_REPO_NAME="aicoe-osc-demo" \
    GIT_REPO_URL="https://github.com/os-climate/aicoe-osc-demo"

USER root

# Adding the poppler utils dependency
RUN yum install -y poppler-utils
RUN yum install -y java-1.8.0-openjdk

# Copying in source code
COPY . /tmp/src
# Change file ownership to the assemble user. Builder image must support chown command.
RUN chown -R 1001:0 /tmp/src
USER 1001
# Assemble script sourced from builder image based on user input or image metadata.
# If this file does not exist in the image, the build will fail.
RUN /opt/app-root/builder/assemble
# Run script sourced from builder image based on user input or image metadata.
# If this file does not exist in the image, the build will fail.

CMD /opt/app-root/builder/run
