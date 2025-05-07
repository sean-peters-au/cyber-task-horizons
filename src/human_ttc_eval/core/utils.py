import re
import logging
from datetime import datetime, timezone # For timestamp parsing, if needed here

# Configure basic logging - this can be enhanced later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def slugify(text: str) -> str:
    """
    Convert a string to a slug.
    Lowercase, replaces spaces and special characters with underscores.
    Handles leading/trailing underscores and multiple underscores.
    """
    if not isinstance(text, str):
        # Or raise TypeError, depending on desired strictness
        logging.warning(f"slugify expected a string, got {type(text)}. Returning empty string.")
        return ""
    text = text.lower()
    text = re.sub(r'\s+', '_', text)          # Replace spaces with underscores
    text = re.sub(r'[^\w_]', '', text)       # Remove non-alphanumeric characters except underscore
    text = re.sub(r'__+', '_', text)          # Replace multiple underscores with single
    text = text.strip('_')                  # Remove leading/trailing underscores
    return text

# Other potential utility functions to be added later:
# - Timestamp parsing helpers (if more complex than datetime.fromisoformat)
# - Common file reading/writing utilities (though Pathlib is quite good)
# - Shared constants if not fitting elsewhere 