#!/usr/bin/env python3
"""
Startup script for the Tetris RL API

This script starts the FastAPI server with proper configuration.
"""

import uvicorn
import os
import sys

def main():
    """Start the API server."""
    print("🚀 Starting Tetris RL API Server")
    print("=" * 50)
    
    # Configuration
    host = "0.0.0.0"
    port = 8000
    
    print(f"🌐 Server will be available at:")
    print(f"   - Local: http://localhost:{port}")
    print(f"   - Network: http://{host}:{port}")
    print(f"   - API Documentation: http://localhost:{port}/docs")
    print(f"   - Alternative Docs: http://localhost:{port}/redoc")
    print("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,  # Auto-reload on code changes
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 