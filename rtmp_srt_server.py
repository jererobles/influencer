#!/usr/bin/env python3
"""
Simple RTMP/SRT Streaming Server

This script implements a basic RTMP/SRT streaming server that allows you to push
streams from OBS or other streaming software and access them via various protocols.
It uses MediaMTX (formerly rtsp-simple-server) which is a multi-protocol
media server that supports RTMP, RTSP, SRT, HLS, and WebRTC.

Requirements:
- Python 3.6+
- internet connection to download MediaMTX binary

Usage:
1. Run this script
2. In OBS, set your streaming URL to rtmp://localhost/live
3. Set your stream key to your preferred stream name (e.g., "mystream")
4. Start streaming in OBS
5. Access your stream:
   - RTMP: rtmp://localhost/live/mystream
   - SRT: srt://localhost:8890?streamid=read:live/mystream
   - WebRTC: http://localhost:8889/live/mystream
   - HLS: http://localhost:8888/live/mystream
"""

import os
import sys
import platform
import shutil
import tempfile
import zipfile
import tarfile
import subprocess
import argparse
import logging
import time
import threading
from pathlib import Path
import urllib.request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('rtmp-srt-server')

# MediaMTX release info
MEDIAMTX_VERSION = "1.12.2"
GITHUB_RELEASE_URL = f"https://github.com/bluenviron/mediamtx/releases/download/v{MEDIAMTX_VERSION}"

def get_system_info():
    """Get the current system's OS and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Normalize architecture names
    if machine in ('x86_64', 'amd64'):
        arch = 'amd64'
    elif machine in ('aarch64', 'arm64'):
        arch = 'arm64'
    elif machine.startswith('arm'):
        arch = 'armv7'
    else:
        arch = machine
        
    return system, arch

def get_mediamtx():
    """Get the MediaMTX binary, downloading only if necessary."""
    system, arch = get_system_info()
    
    # Map system name to MediaMTX release name
    if system == 'linux':
        os_name = 'linux'
    elif system == 'darwin':
        os_name = 'darwin'
    elif system == 'windows':
        os_name = 'windows'
    else:
        raise ValueError(f"Unsupported operating system: {system}")

    # Determine expected executable name
    if system == 'windows':
        executable_name = "mediamtx.exe"
    else:
        executable_name = "mediamtx"
        
    # Use a persistent location for storing the binary
    bin_dir = Path.home() / ".mediamtx" / MEDIAMTX_VERSION
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    executable_path = bin_dir / executable_name
    
    # Check if the executable already exists
    if executable_path.exists() and os.access(executable_path, os.X_OK):
        logger.info(f"MediaMTX binary already exists at {executable_path}")
        return executable_path
    
    # If not, download it
    logger.info(f"MediaMTX binary not found, downloading version {MEDIAMTX_VERSION}...")
    
    # Construct filename
    if system == 'windows':
        filename = f"mediamtx_v{MEDIAMTX_VERSION}_{os_name}_{arch}.zip"
    else:
        filename = f"mediamtx_v{MEDIAMTX_VERSION}_{os_name}_{arch}.tar.gz"
        
    url = f"{GITHUB_RELEASE_URL}/{filename}"
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    download_path = temp_dir / filename
    
    # Download the file
    logger.info(f"Downloading MediaMTX {MEDIAMTX_VERSION} for {os_name}_{arch} from {url}")
    try:
        urllib.request.urlretrieve(url, download_path)
    except Exception as e:
        logger.error(f"Failed to download MediaMTX: {e}")
        raise
    
    # Extract the archive
    extract_dir = temp_dir / "extract"
    extract_dir.mkdir()
    
    try:
        if system == 'windows':
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            with tarfile.open(download_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
    except Exception as e:
        logger.error(f"Failed to extract MediaMTX archive: {e}")
        raise
    
    # Find the executable
    if system == 'windows':
        extracted_executable = extract_dir / executable_name
    else:
        extracted_executable = extract_dir / executable_name
    
    if not extracted_executable.exists():
        # Look for the executable in subdirectories
        candidates = list(extract_dir.glob(f"**/{executable_name}*"))
        if not candidates:
            raise FileNotFoundError("MediaMTX executable not found in the downloaded archive")
        extracted_executable = candidates[0]
    
    # Copy to the persistent location
    shutil.copy2(extracted_executable, executable_path)
    
    # Ensure the file is executable
    if system != 'windows':
        executable_path.chmod(executable_path.stat().st_mode | 0o111)
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    logger.info(f"MediaMTX binary installed at {executable_path}")
    return executable_path

def create_config(path, rtmp_port=1935, srt_port=8890, webrtc_port=8889, hls_port=8888, api_port=9997):
    """Create MediaMTX configuration file."""
    config = f"""# MediaMTX configuration for simple RTMP/SRT server
logLevel: info
logDestinations: [stdout]

api: yes
apiAddress: :{api_port}
metrics: yes
metricsAddress: :9998
pprof: yes
pprofAddress: :9999

rtspAddress: :8554
rtspsAddress: :8322
rtpAddress: :8000
rtcpAddress: :8001
rtmpAddress: :{rtmp_port}
hlsAddress: :{hls_port}
webrtcAddress: :{webrtc_port}
srtAddress: :{srt_port}

# Path configuration
paths:
  # Default path for live streaming
  live:
    source: publisher
    
    # Record streams to disk (disabled by default)
    record: no
    # recordPath: ./recordings/%path/%Y-%m-%d_%H-%M-%S-%f

    # Command to run when the stream is ready to be read
    runOnReady: |
      #!/bin/sh
      # Notify the main application about the new stream
      echo "STREAM_READY:$MTX_PATH:$MTX_SOURCE_TYPE:$MTX_SOURCE_ID" > /tmp/mediamtx_notify

  # Catch-all for all other paths
  all_others:
"""
    with open(path, 'w') as f:
        f.write(config)
    return path

def run_server(executable, config_path, verbose=False):
    """Run the MediaMTX server."""
    cmd = [str(executable), str(config_path)]
    
    try:
        logger.info(f"Starting MediaMTX server")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
            universal_newlines=True
        )
        
        # Check if the process started successfully
        time.sleep(1)
        if process.poll() is not None:
            logger.error(f"MediaMTX server failed to start (exit code {process.returncode})")
            if not verbose:
                # Print output to help diagnose the issue
                output, _ = process.communicate()
                logger.error(f"Server output: {output}")
            return process.returncode
        
        # Process started successfully, now handle it
        if verbose:
            logger.info("MediaMTX server is running. Press Ctrl+C to stop.")
            try:
                # Just wait for the process to end
                process.wait()
            except KeyboardInterrupt:
                logger.info("Stopping server...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                logger.info("Server stopped.")
        else:
            logger.info("MediaMTX server is running in the background.")
            logger.info("You can now push your OBS stream to rtmp://localhost/live")
            logger.info("Access your stream:")
            logger.info("  - RTMP: rtmp://localhost/live/mystream")
            logger.info("  - RTSP: rtsp://localhost:8554/live/mystream")
            logger.info("  - SRT: srt://localhost:8890?streamid=read:live/mystream")
            logger.info("  - WebRTC: http://localhost:8889/live/mystream")
            logger.info("  - HLS: http://localhost:8888/live/mystream")
            logger.info("")
            logger.info("Press Ctrl+C to stop the server")
            
            # Start a thread to read and log the output
            def log_output():
                while process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        logger.debug(line.strip())
            
            if not verbose:
                threading.Thread(target=log_output, daemon=True).start()
            
            try:
                # Keep the main thread running as long as the server is running
                while process.poll() is None:
                    time.sleep(0.5)
                
                # If we get here, the server has stopped on its own
                exit_code = process.returncode
                logger.error(f"MediaMTX server has stopped unexpectedly (exit code {exit_code})")
                return exit_code
                
            except KeyboardInterrupt:
                logger.info("Stopping server...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                logger.info("Server stopped.")
        
        return process.returncode
    except Exception as e:
        logger.error(f"Failed to run MediaMTX server: {e}")
        raise

def main():
    """Main function to handle command-line arguments and run the server."""
    parser = argparse.ArgumentParser(description="Simple RTMP/SRT Streaming Server")
    parser.add_argument("-c", "--config", type=str, help="Path to custom config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--rtmp-port", type=int, default=1935, help="RTMP port (default: 1935)")
    parser.add_argument("--srt-port", type=int, default=8890, help="SRT port (default: 8890)")
    parser.add_argument("--webrtc-port", type=int, default=8889, help="WebRTC port (default: 8889)")
    parser.add_argument("--hls-port", type=int, default=8888, help="HLS port (default: 8888)")
    parser.add_argument("--api-port", type=int, default=9997, help="API port (default: 9997)")
    
    args = parser.parse_args()
    
    try:
        # Set log level to DEBUG if verbose is enabled
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Get MediaMTX binary (download only if needed)
        executable = get_mediamtx()
        
        # Create config file if not provided
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = Path.cwd() / "mediamtx.yml"
            create_config(
                config_path,
                rtmp_port=args.rtmp_port,
                srt_port=args.srt_port,
                webrtc_port=args.webrtc_port,
                hls_port=args.hls_port,
                api_port=args.api_port
            )
        
        # Run server - this will block until Ctrl+C is pressed or the server exits
        return run_server(executable, config_path, verbose=args.verbose)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())