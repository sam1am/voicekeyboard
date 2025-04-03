#!/usr/bin/env python3
import os
import signal

pid_file = "./tmp/voice_command_processor.pid"

def stop_service():
    """Stop the running voice command processor service"""
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            # Send termination signal
            os.kill(pid, signal.SIGTERM)
            print(f"Sent termination signal to Voice Command Processor (PID: {pid})")
        except (ProcessLookupError, ValueError):
            print("Voice Command Processor is not running")
            if os.path.exists(pid_file):
                os.remove(pid_file)
        except Exception as e:
            print(f"Error stopping service: {e}")
    else:
        print("Voice Command Processor is not running")

if __name__ == "__main__":
    stop_service()