"""Network-related utilities for the NYUCTF benchmark.

Handles the generation of scripts and configuration files used to set up
the sandboxed Docker environment's network restrictions.
"""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# This script is written to a file and sourced by the agent's shell.
# It handles the one-time setup of GPG keys and network restrictions.
SETUP_SCRIPT_CONTENT = '''
#!/bin/bash
# (c) 2024, Human TTC Eval Team
#
# One-time setup script for NYUCTF evaluation environment.
# This script is sourced by interactive bash shells and runs only once.

# Exit if setup has already been completed or if not an interactive shell
if [ -f /tmp/nyuctf_setup_complete ] || [[ "$-" != *i* ]]; then
    return
fi

echo "--- NYUCTF Environment Setup (One-Time) ---"

# Temporarily open firewall for setup
iptables -P OUTPUT ACCEPT 2>/dev/null || true

# --- 1. Install Dependencies & Fix GPG ---
# This is the critical part. We need to be very explicit to get
# an old system's GPG keys updated.
echo "[1/4] Fixing GPG keys and updating package lists..."
export DEBIAN_FRONTEND=noninteractive
# Step 1: Force add the new Kali signing key.
wget -q -O - https://archive.kali.org/archive-key.asc | apt-key add -
# Step 2: Now update package lists, allowing untrusted sources just this once.
apt-get update --allow-insecure-repositories --allow-unauthenticated
# Step 3: Install the new keyring package to make the trust permanent.
apt-get install -y --allow-unauthenticated kali-archive-keyring
# Step 4: A final update should now succeed without any special flags.
apt-get update
echo "      Done."

# --- 2. Create iptables Lockdown Function ---
echo "[2/4] Creating network lockdown function..."
cat << 'EOF' > /usr/local/bin/apply-nyuctf-lockdown
#!/bin/bash
iptables -F OUTPUT 2>/dev/null || true
iptables -P OUTPUT DROP 2>/dev/null || true
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
EOF
chmod +x /usr/local/bin/apply-nyuctf-lockdown
echo "      Done."

# --- 3. Create apt/apt-get Wrappers ---
echo "[3/4] Creating wrappers for apt and apt-get..."
if [ ! -f /usr/bin/apt-get.real ]; then
    mv /usr/bin/apt-get /usr/bin/apt-get.real
fi
cat << 'EOF' > /usr/bin/apt-get
#!/bin/bash
# Silently open firewall for apt, then lock it down again.
iptables -P OUTPUT ACCEPT 2>/dev/null
/usr/bin/apt-get.real "$@"
/usr/local/bin/apply-nyuctf-lockdown
EOF
chmod +x /usr/bin/apt-get

if [ -f /usr/bin/apt ] && [ ! -f /usr/bin/apt.real ]; then
    mv /usr/bin/apt /usr/bin/apt.real
    ln -s /usr/bin/apt-get /usr/bin/apt
fi
echo "      Done."

# --- 4. Apply Initial Lockdown ---
echo "[4/4] Applying initial network lockdown..."
/usr/local/bin/apply-nyuctf-lockdown
echo "      Done."

# Mark setup as complete
touch /tmp/nyuctf_setup_complete
echo "--- Setup Complete. Network is now restricted. ---"
'''

def create_setup_script() -> Path:
    """
    Writes the one-time setup script to a temporary file.

    Returns:
        The path to the executable temporary script file.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(SETUP_SCRIPT_CONTENT)
        temp_file = Path(f.name)

    temp_file.chmod(0o755)
    logger.debug(f"Created one-time setup script: {temp_file}")
    return temp_file 