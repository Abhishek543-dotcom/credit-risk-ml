"""
Script to run the Credit Risk API with monitoring
"""

import uvicorn
import yaml
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run the API server"""
    parser = argparse.ArgumentParser(description='Run Credit Risk API')
    parser.add_argument('--host', type=str, default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Get host and port
    host = args.host or config['api']['host']
    port = args.port or config['api']['port']

    logger.info("=" * 60)
    logger.info("Starting Credit Risk ML API")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Docs: http://{host}:{port}/docs")
    logger.info(f"ReDoc: http://{host}:{port}/redoc")
    logger.info("=" * 60)

    # Run server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
