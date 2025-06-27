"""Docker-related utilities for the NYUCTF benchmark.

Handles the generation and manipulation of Docker Compose files.
"""
import logging
import socket
import subprocess
import tempfile
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from . import config as nyuctf_config

logger = logging.getLogger(__name__)


def get_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def extract_category_from_path(compose_file_path: Path) -> Optional[str]:
    """
    Extract the challenge category from the file path.
    e.g. .../pwn/my_first_pwnie/docker-compose.yml -> 'pwn'
    """
    try:
        if len(compose_file_path.parts) >= 3:
            category = compose_file_path.parts[-3].lower()
            if category in nyuctf_config.CTF_CATEGORIES:
                logger.debug(f"Extracted category '{category}' from {compose_file_path}")
                return category
        logger.warning(f"Could not extract valid category from path: {compose_file_path}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting category from path {compose_file_path}: {e}")
        return None


def _get_ctfnet_subnet() -> Optional[str]:
    """Inspects the ctfnet network on the host to get its subnet."""
    try:
        result = subprocess.run(
            ["docker", "network", "inspect", nyuctf_config.NETWORK_NAME],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        network_data = json.loads(result.stdout)
        if network_data and network_data[0].get("IPAM", {}).get("Config"):
            subnet = network_data[0]["IPAM"]["Config"][0].get("Subnet")
            if subnet:
                logger.info(f"Dynamically found ctfnet subnet: {subnet}")
                return subnet
        logger.warning("Could not find subnet in ctfnet network configuration.")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, IndexError) as e:
        logger.error(f"Failed to inspect ctfnet network, firewall may be incomplete: {e}")
        return None


def modify_compose_file(original_compose_file: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Modifies a docker-compose.yml file for the evaluation environment.

    - Adds a default agent service with CTF tools and network restrictions.
    - Adds network aliases to challenge services.
    - Replaces fixed ports with dynamic ones to avoid conflicts.

    Returns:
        A tuple containing the path to the modified compose file and port info.
    """
    try:
        with open(original_compose_file, 'r') as f:
            compose_data = yaml.safe_load(f) or {'services': {}}
    except FileNotFoundError:
        compose_data = {'services': {}}

    port_info = {}
    
    # Get the ctfnet subnet to pass to the agent
    ctfnet_subnet = _get_ctfnet_subnet()

    # 1. Add agent service
    _add_agent_service(compose_data, ctfnet_subnet)

    # 2. Process existing challenge services
    category = extract_category_from_path(original_compose_file)
    for service_name, service_config in compose_data.get('services', {}).items():
        if service_name == 'default':
            # This is our custom agent, ensure platform is set for ARM hosts
            service_config['platform'] = 'linux/amd64'
            continue
        
        # Add network alias
        _add_network_alias(service_config, category)
        
        # Ensure platform compatibility for ARM hosts
        if 'platform' not in service_config:
            service_config['platform'] = 'linux/amd64'
        
        # Remap ports and get info
        service_port_info = _remap_service_ports(service_config)
        if service_port_info:
            port_info[service_name] = service_port_info[0]

        # Convert relative build contexts to absolute paths
        _absolutize_build_context(service_config, original_compose_file)

    # 3. Ensure ctfnet network exists
    if 'networks' not in compose_data:
        compose_data['networks'] = {}
    if nyuctf_config.NETWORK_NAME not in compose_data['networks']:
        compose_data['networks'][nyuctf_config.NETWORK_NAME] = {'external': True}

    # 4. Save to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.safe_dump(compose_data, f, default_flow_style=False)
        temp_file = Path(f.name)

    logger.debug(f"Created modified compose file: {temp_file}")
    logger.debug(f"Port mappings: {port_info}")
    return temp_file, port_info


def create_minimal_compose_file() -> Path:
    """Creates a compose file with only the agent service."""
    compose_data = {'services': {}}
    
    ctfnet_subnet = _get_ctfnet_subnet()
    _add_agent_service(compose_data, ctfnet_subnet)
    
    # Also specify platform for the minimal service
    if 'default' in compose_data['services']:
        compose_data['services']['default']['platform'] = 'linux/amd64'

    compose_data['networks'] = {
        nyuctf_config.NETWORK_NAME: {'external': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.safe_dump(compose_data, f, default_flow_style=False)
        temp_file = Path(f.name)

    logger.debug(f"Created minimal compose file for static challenge: {temp_file}")
    return temp_file


def _add_agent_service(compose_data: dict, ctfnet_subnet: Optional[str]) -> None:
    """Adds the agent service configuration to the compose data."""
    if 'services' not in compose_data:
        compose_data['services'] = {}
        
    # Add a specific firewall rule for the ctfnet subnet if found
    ctfnet_rule = f"iptables -A OUTPUT -d {ctfnet_subnet} -j ACCEPT" if ctfnet_subnet else ""

    # This command runs inside the container to set up the firewall and keep it alive.
    setup_command = f"""bash -c '
        cat << "LOCK" > /usr/local/bin/apply-nyuctf-lockdown
#!/bin/bash
iptables -F OUTPUT 2>/dev/null || true
iptables -P OUTPUT DROP 2>/dev/null || true
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
# Allow standard Docker bridge network
iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT
# Dynamically add rule for the ctfnet network
{ctfnet_rule}
# Allow DNS
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
LOCK
        chmod +x /usr/local/bin/apply-nyuctf-lockdown
        /usr/local/bin/apply-nyuctf-lockdown
        
        export PATH="/usr/local/bin:$PATH"
        mkdir -p /usr/local/bin
        
        printf "%s\\n" "#!/bin/bash" "iptables -P OUTPUT ACCEPT 2>/dev/null" "exec /usr/bin/apt-get \\"\\$@\\"" "/usr/local/bin/apply-nyuctf-lockdown" > /usr/local/bin/apt-get
        chmod +x /usr/local/bin/apt-get
        
        while true; do sleep 3600; done
    '"""

    compose_data['services']['default'] = {
        'image': nyuctf_config.AGENT_IMAGE,
        'x-local': True,  # Ensures inspect_ai uses the local image, not pulls from a remote registry
        'platform': 'linux/amd64', # Explicitly set platform for our agent on ARM hosts
        'init': True,
        'networks': [nyuctf_config.NETWORK_NAME],
        'privileged': True,
        'cap_add': ['NET_ADMIN', 'SYS_PTRACE'],
        'command': setup_command
    }
    
def _add_network_alias(service_config: dict, category: Optional[str]) -> None:
    """Adds a category-based network alias to a service."""
    if not category:
        return
        
    if 'networks' not in service_config:
        service_config['networks'] = {}
    
    # Handle if networks is a list (e.g., ['default'])
    if isinstance(service_config['networks'], list):
        networks_dict = {net_name: {} for net_name in service_config['networks']}
        service_config['networks'] = networks_dict

    if nyuctf_config.NETWORK_NAME not in service_config['networks']:
        service_config['networks'][nyuctf_config.NETWORK_NAME] = {}

    alias = f"{category}.chal.csaw.io"
    aliases = service_config['networks'][nyuctf_config.NETWORK_NAME].get('aliases', [])
    if alias not in aliases:
        aliases.append(alias)
    service_config['networks'][nyuctf_config.NETWORK_NAME]['aliases'] = aliases
    logger.debug(f"Added network alias '{alias}'")

def _remap_service_ports(service_config: dict) -> list:
    """Remaps fixed ports to dynamic ones, returns port info."""
    if 'ports' not in service_config:
        return []

    new_ports = []
    service_port_info = []
    for port_mapping in service_config.get('ports', []):
        if isinstance(port_mapping, str) and ':' in port_mapping:
            host_port, container_port = port_mapping.split(':', 1)
            if host_port.isdigit():
                free_port = get_free_port()
                new_ports.append(f"{free_port}:{container_port}")
                service_port_info.append({'host_port': free_port, 'container_port': int(container_port)})
                logger.debug(f"Remapped port {port_mapping} -> {free_port}:{container_port}")
            else:
                new_ports.append(port_mapping)
        elif isinstance(port_mapping, int):
            free_port = get_free_port()
            new_ports.append(f"{free_port}:{port_mapping}")
            service_port_info.append({'host_port': free_port, 'container_port': port_mapping})
            logger.debug(f"Remapped port {port_mapping} -> {free_port}:{port_mapping}")
        else:
            new_ports.append(port_mapping)
    
    service_config['ports'] = new_ports
    return service_port_info


def _absolutize_build_context(service_config: dict, original_compose_file: Path):
    """Converts relative build context paths to absolute paths."""
    if 'build' not in service_config:
        return

    build_config = service_config['build']
    if isinstance(build_config, str):
        abs_path = str(original_compose_file.parent / build_config)
        service_config['build'] = abs_path
        logger.debug(f"Converted build path to absolute: {abs_path}")
    elif isinstance(build_config, dict) and 'context' in build_config:
        rel_context = build_config['context']
        abs_context = str(original_compose_file.parent / rel_context)
        build_config['context'] = abs_context
        logger.debug(f"Converted build context to absolute: {abs_context}") 