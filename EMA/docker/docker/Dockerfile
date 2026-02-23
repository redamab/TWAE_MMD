# TWAE_MMD VS Code Docker Environment
# Optimized for VS Code development with TensorFlow/Keras and GPU support (RTX 2060)
# All packages tested and compatible with Python 3.8 (TensorFlow base image)

FROM tensorflow/tensorflow:2.13.0-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /workspace/TWAE_AMP_Generation

# Install system dependencies including VS Code Server requirements
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tree \
    graphviz \
    openssh-server \
    sudo \
    locales \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create a non-root user for VS Code
RUN useradd -m -s /bin/bash -G sudo vscode && \
    echo "vscode:vscode" | chpasswd && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install VS Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Upgrade pip and install Python packages
RUN pip install --upgrade pip

# Install core scientific packages - PYTHON 3.8 COMPATIBLE VERSIONS
RUN pip install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    joblib==1.3.2

# Install TensorFlow ecosystem
RUN pip install \
    tensorflow-probability==0.21.0 \
    tensorboard==2.13.0 \
    keras==2.13.1

# Install optimization packages
RUN pip install \
    optuna==3.3.0 \
    bayesian-optimization==1.4.3 \
    hyperopt==0.2.7

# Install biological analysis packages
RUN pip install \
    biopython==1.81 \
    modlamp==4.3.0 \
    peptides==0.3.0

# Install visualization packages - PYTHON 3.8 COMPATIBLE
RUN pip install \
    plotly==5.15.0 \
    bokeh==3.1.1 \
    umap-learn==0.5.3

# Install ML/NLP packages - PYTHON 3.8 COMPATIBLE
RUN pip install \
    transformers==4.30.0 \
    datasets==2.13.0 \
    tokenizers==0.13.3

# Install configuration and utility packages - PYTHON 3.8 COMPATIBLE
RUN pip install \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    pydantic==1.10.12 \
    rich==13.4.2 \
    typer==0.9.0 \
    python-dotenv==1.0.0 \
    colorama==0.4.6 \
    psutil==5.9.5

# Install additional scientific packages - PYTHON 3.8 COMPATIBLE
RUN pip install \
    statsmodels==0.14.0 \
    networkx==3.1 \
    h5py==3.9.0

# Install development packages - PYTHON 3.8 COMPATIBLE
RUN pip install \
    pytest==7.4.0 \
    black==23.7.0 \
    flake8==6.0.0 \
    mypy==1.4.1 \
    pre-commit==3.3.3

# Install Jupyter for notebook support in VS Code - PYTHON 3.8 COMPATIBLE
RUN pip install \
    jupyter==1.0.0 \
    jupyterlab==4.0.5 \
    notebook==6.5.4 \
    ipykernel==6.25.0

# Install VS Code extensions for Python development
RUN code-server --install-extension ms-python.python && \
    code-server --install-extension ms-toolsai.jupyter && \
    code-server --install-extension ms-azuretools.vscode-docker && \
    code-server --install-extension ms-python.black-formatter && \
    code-server --install-extension ms-python.flake8 && \
    code-server --install-extension ms-python.mypy-type-checker

# Create necessary directories
RUN mkdir -p /workspace/TWAE_AMP_Generation/{src,scripts,tests,data,models,results,logs,docs,environment} && \
    chown -R vscode:vscode /workspace

# Copy project files
COPY . /workspace/TWAE_AMP_Generation/
RUN chown -R vscode:vscode /workspace/TWAE_AMP_Generation

# Set Python path
ENV PYTHONPATH=/workspace/TWAE_AMP_Generation

# Expose ports for VS Code Server and TensorBoard
EXPOSE 8080 6006 8888

# Create VS Code configuration directory
RUN mkdir -p /home/vscode/.local/share/code-server/User && \
    chown -R vscode:vscode /home/vscode/.local

# Create VS Code settings
RUN echo '{\n\
    "python.defaultInterpreterPath": "/usr/bin/python3",\n\
    "python.linting.enabled": true,\n\
    "python.linting.pylintEnabled": false,\n\
    "python.linting.flake8Enabled": true,\n\
    "python.formatting.provider": "black",\n\
    "python.formatting.blackArgs": ["--line-length", "88"],\n\
    "jupyter.askForKernelRestart": false,\n\
    "files.autoSave": "afterDelay",\n\
    "terminal.integrated.shell.linux": "/bin/bash",\n\
    "workbench.colorTheme": "Default Dark+",\n\
    "editor.fontSize": 14,\n\
    "editor.tabSize": 4,\n\
    "editor.insertSpaces": true\n\
}' > /home/vscode/.local/share/code-server/User/settings.json

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 TWAE_MMD VS Code Environment Starting..."\n\
echo "🎯 TensorFlow: $(python -c \"import tensorflow as tf; print(tf.__version__)\")"\n\
echo "🧬 GPU Available: $(python -c \"import tensorflow as tf; print(len(tf.config.list_physical_devices(\\\"GPU\\\")) > 0)\")"\n\
echo "💻 Starting VS Code Server on port 8080..."\n\
echo "📈 TensorBoard available on port 6006"\n\
echo "🔗 Access VS Code at: http://localhost:8080"\n\
echo "🔑 Password: twae_mmd_research"\n\
echo "🚀 Ready for TWAE_MMD development!"\n\
su - vscode -c "code-server --bind-addr 0.0.0.0:8080 --auth password --password twae_mmd_research /workspace/TWAE_AMP_Generation"\n\
' > /usr/local/bin/start-vscode.sh && chmod +x /usr/local/bin/start-vscode.sh

# --- START UID/GID FIX for Host Permissions ---
# Your Host UID and GID are 1002
ARG HOST_UID=1002
ARG HOST_GID=1002
# Modify the 'vscode' user inside the container to match your host UID/GID
RUN usermod -u ${HOST_UID} vscode && groupmod -g ${HOST_GID} vscode
# --- END UID/GID FIX ---

# Switch to vscode user
USER vscode

# Default command
CMD ["/usr/local/bin/start-vscode.sh"]

