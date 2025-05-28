#!/usr/bin/env python3
"""
Tetris RL API Server Startup Script
"""

import uvicorn
import os
import sys

def main():
    """Start the API server."""
    print("🚀 Starting Tetris RL API Server")
    print("=" * 50)
    
    print("🌐 Server will be available at:")
    print("   - Local: http://localhost:8000")
    print("   - Network: http://0.0.0.0:8000")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Alternative Docs: http://localhost:8000/redoc")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 