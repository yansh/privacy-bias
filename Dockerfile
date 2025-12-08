FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl sudo bzip2 ca-certificates git build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set Conda environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Accept Conda TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Set up Conda environment
RUN conda init bash
RUN conda create -n myenv python=3.11
RUN conda install -n myenv ipykernel

# Copy environment.yml and update environment
COPY environment.yml .
RUN conda run -n myenv conda env update --file environment.yml && conda clean --all

# Create auditor user safely
RUN getent group auditorgroup || addgroup --gid 1000 auditorgroup && \
    id -u auditor || adduser --disabled-password --gecos "" --uid 1000 --gid 1000 auditor && \
    mkdir -p /home/auditor && chown -R auditor:auditorgroup /home/auditor

# Now you can modify the auditor user's .bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/auditor/.bashrc && \
    echo "conda activate myenv" >> /home/auditor/.bashrc && \
    chown auditor:auditorgroup /home/auditor/.bashrc

# Give sudo access to auditor
RUN usermod -aG sudo auditor && \
    echo "auditor ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/auditor

# Switch to auditor environment
USER auditor
ENV HOME=/home/auditor
WORKDIR /home/auditor

EXPOSE 8888

CMD ["conda", "run", "-n", "myenv", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
