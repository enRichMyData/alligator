FROM jupyter/base-notebook:latest

# Copy project configuration and requirements from the repository root
COPY pyproject.toml .
COPY ./alligator ./alligator

# Install Python dependencies
RUN PIP_DEFAULT_TIMEOUT=100 pip install --no-cache-dir -e "."

# Download spaCy model
RUN python -m spacy download en_core_web_sm && python -m spacy download en_core_web_trf


# Expose Jupyter Notebook port
EXPOSE 8888
