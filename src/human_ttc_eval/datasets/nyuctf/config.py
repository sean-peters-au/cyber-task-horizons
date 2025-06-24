"""Configuration constants for the NYUCTF benchmark."""

# --- General ---
AGENT_IMAGE = "aisiuk/evals-cybench-agent-sandbox:0.1"
NETWORK_NAME = "ctfnet"
COMMAND_TIMEOUT = 300
CTF_CATEGORIES = ['pwn', 'crypto', 'web', 'misc', 'forensics', 'rev']

# --- GPG Fix ---
GPG_KEY_URL = "https://archive.kali.org/archive-key.asc"
KEYRING_PACKAGE = "kali-archive-keyring"

# --- Network Restrictions ---
BLOCKED_DOMAINS = [
    'www.google.com', 'google.com',
    'www.bing.com', 'bing.com',
    'www.duckduckgo.com', 'duckduckgo.com',
    'www.yahoo.com', 'yahoo.com', 'search.yahoo.com',
    'www.baidu.com', 'baidu.com',
    'www.yandex.com', 'yandex.com', 'yandex.ru',
    'github.com', 'gist.github.com', 'raw.githubusercontent.com',
    'ctftime.org', 'www.ctftime.org',
    'writeup.ctf.link',
    'ctf-wiki.org',
    'medium.com',
    'reddit.com', 'www.reddit.com',
    'stackoverflow.com',
    'youtube.com', 'www.youtube.com',
    'pastebin.com',
    'gitlab.com',
] 