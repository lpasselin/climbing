
FROM --platform=linux/amd64 ubuntu:20.04 

ARG AZURE_CLI_VERSION=2.35.0

# install all packages with apt here.
# Also installs azure-cli here
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git git-crypt \
    libyaml-dev \
    wget \
    nano \
    gcc \
    g++ \
    fuse \
    dh-make \
    ca-certificates curl apt-transport-https lsb-release gnupg && \
    curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null && \
    AZ_REPO=$(lsb_release -cs) && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | tee /etc/apt/sources.list.d/azure-cli.list && \
    apt-get update && apt-get install azure-cli=$AZURE_CLI_VERSION-1~$AZ_REPO && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# install mini conda
ENV CONDA_DIR=/opt/miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_ENV_PATH "${CONDA_DIR}/envs/projectenv"

COPY environment.yml environment.yml

RUN conda env create --prefix ${CONDA_ENV_PATH} -f environment.yml && \
    conda clean -afy && \
    pip cache purge

# Setup bash to automatically ativate project env and login to do so
RUN echo "source activate ${CONDA_ENV_PATH}" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
