import os
import logging
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from tqdm import tqdm

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install playwright tqdm")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)
