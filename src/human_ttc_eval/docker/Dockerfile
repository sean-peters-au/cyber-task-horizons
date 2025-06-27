FROM --platform=linux/amd64 aisiuk/evals-cybench-agent-sandbox:0.1

# -----------------------------------------------------------------------------
# GPG Key and Package Fixes
# -----------------------------------------------------------------------------
# This section permanently fixes the outdated GPG keys in the base image
# and ensures apt-get works reliably.

# Set non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Run all commands in a single layer to optimize build time and layers
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated wget && \
    wget -q -O - https://archive.kali.org/archive-key.asc | apt-key add - && \
    apt-get update --allow-insecure-repositories --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated kali-archive-keyring && \
    apt-get update 

# -----------------------------------------------------------------------------
# Common Tools
# -----------------------------------------------------------------------------
# Install common CTF tools that models frequently use but may not be in the
# base image. This includes mtools for FAT filesystem analysis.
RUN apt-get install -y mtools strace gdb ltrace bsdmainutils binwalk

# -----------------------------------------------------------------------------
# Python Environment
# -----------------------------------------------------------------------------
# Install pip and common Python packages.
# pycryptodome provides the 'Crypto' namespace needed for some challenges.
RUN apt-get install -y python3-pip python3-dev libffi-dev build-essential && \
    pip install --no-cache-dir pycryptodome pwntools gmpy2 z3-solver 

    
# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Revert to a standard shell prompt
ENV PS1='\\u@\\h:\\w\\$ ' 

WORKDIR /root/

# Keep the container running indefinitely.
CMD ["tail", "-f", "/dev/null"]