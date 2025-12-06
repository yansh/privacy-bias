FROM python:3.13.6-trixie

# Set environment variables for Conda installation
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Accept the Conda TOS automatically
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Set up Conda environment
RUN conda init bash
RUN conda create -n myenv python=3.13

RUN conda install -n myenv ipykernel

# Activate the environment by default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install required packages using Conda (replace with the appropriate packages or your environment.yml file)
COPY environment.yml .  

RUN conda env update --file environment.yml && conda clean --all

RUN addgroup --gid 1000 auditorgroup && \
    adduser --disabled-password --gecos "" --uid 1000 --gid 1000 auditor && \
    mkdir -p /home/auditor && \
    chown -R auditor:auditorgroup /home/auditor

RUN conda install -n myenv -c conda-forge jupyterlab



USER auditor
ENV HOME=/home/auditor
WORKDIR /home/auditor

EXPOSE 8888

# Start JupyterLab when the container runs
CMD ["conda", "run", "-n", "myenv", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=''"]