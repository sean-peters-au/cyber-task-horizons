# Use a standard Python base image
FROM python:3.11-slim-bookworm

# Install pytest and inspect-tool-support globally
RUN pip install --no-cache-dir pytest inspect-tool-support \
    && inspect-tool-support post-install --no-web-browser

# Set up a non-root user
RUN useradd --create-home --shell /bin/bash agent

# Switch to the non-root user and set working directory
USER agent