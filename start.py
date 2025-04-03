#!/usr/bin/env python3
import os
import time
import signal
import sys
import threading
import tempfile
import json
import base64
import wave
import requests
import daemon
from dotenv import load_dotenv
from pynput import keyboard
from pynput.keyboard import Key, Controller
import sounddevice as sd
import numpy as np
import soundfile as sf

# Global variables
recording = False
audio_frames = []
kb_controller = Controller()
running = True
pid_file = "./tmp/voice_command_processor.pid"

def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    print("Shutting down voice command processor...")
    running = False
    cleanup()
    sys.exit(0)

def cleanup():
    """Clean up resources before exiting"""
    if os.path.exists(pid_file):
        os.remove(pid_file)

def load_system_prompt():
    """Load system prompt from markdown file"""
    try:
        with open("system_prompt.md", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: system_prompt.md not found")
        return ""

def on_press(key):
    """Callback function when a key is pressed"""
    global recording, audio_frames
    
    try:
        # Check if the pressed key is our trigger key
        if hasattr(key, 'char') and key.char == TRIGGER_KEY:
            if not recording:
                # Start recording
                print("Recording started...")
                recording = True
                audio_frames = []
                threading.Thread(target=record_audio).start()
            else:
                # Stop recording and process
                print("Recording stopped. Processing...")
                recording = False
    except AttributeError:
        pass

def record_audio():
    """Record audio until recording is set to False"""
    global recording, audio_frames
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while recording and running:
            time.sleep(0.1)
    
    if audio_frames and running:
        # Convert audio frames to numpy array
        audio_data = np.concatenate(audio_frames, axis=0)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, SAMPLE_RATE)
        
        # Process the recording
        process_recording(temp_file.name)
        
        # Clean up
        temp_file.close()
        os.unlink(temp_file.name)

def audio_callback(indata, frames, time, status):
    """Callback for audio recording"""
    if recording:
        audio_frames.append(indata.copy())

def process_recording(audio_file_path):
    """Process the recorded audio file"""
    # Step 1: Transcribe the audio
    transcription = transcribe_audio(audio_file_path)
    print(f"Transcription: {transcription}")
    
    # Step 2: Send to LLM for processing
    if transcription:
        llm_response = process_with_llm(transcription)
        print(f"LLM Response: {llm_response}")
        
        # Step 3: Execute the command
        execute_command(llm_response)

def transcribe_audio(audio_file_path):
    """Transcribe audio using DeepInfra API"""
    try:
        # Read audio file as base64
        with open(audio_file_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
        
        # API call to DeepInfra
        headers = {
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "audio": audio_data
        }
        response = requests.post(
            f"https://api.deepinfra.com/v1/inference/{SPEECH_MODEL}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            print(f"Transcription error: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return ""

def process_with_llm(text):
    """Process text with LLM to determine action"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = load_system_prompt()
        if not system_prompt:
            return ""
        
        # Format the input prompt based on Llama-3 requirements
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{system_prompt}

Here is the spoken command: "{text}"

Return ONLY the formatted keyboard command with no explanations.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        payload = {
            "input": prompt,
            "stop": ["<|eot_id|>"]
        }
        
        response = requests.post(
            f"https://api.deepinfra.com/v1/inference/{LLM_MODEL}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json().get("results", [{}])[0].get("generated_text", "").strip()
        else:
            print(f"LLM error: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print(f"Error in process_with_llm: {e}")
        return ""

def execute_command(command):
    """Execute the keyboard command"""
    if not command:
        return
    
    # First, delete the trigger keys that were typed
    # Press backspace twice to delete the "&&" characters
    kb_controller.press(Key.backspace)
    kb_controller.release(Key.backspace)
    kb_controller.press(Key.backspace)
    kb_controller.release(Key.backspace)
    
    # Process sequences in parentheses as key combinations
    i = 0
    while i < len(command):
        if command[i] == '(':
            # Find the closing parenthesis
            j = command.find(')', i)
            if j == -1:
                j = len(command)
            
            # Extract the key combination
            combo = command[i+1:j]
            execute_key_combination(combo)
            i = j + 1
        elif command[i:i+1] == '<':
            # Find the closing bracket
            j = command.find('>', i)
            if j == -1:
                j = len(command)
            
            # Extract the special key
            special_key = command[i+1:j]
            press_special_key(special_key)
            i = j + 1
        else:
            # Type the character
            kb_controller.press(command[i])
            kb_controller.release(command[i])
            i += 1
    
    print("Command executed.")

def execute_key_combination(combo):
    """Execute a key combination"""
    keys = []
    i = 0
    while i < len(combo):
        if combo[i:i+1] == '<':
            # Find the closing bracket
            j = combo.find('>', i)
            if j == -1:
                j = len(combo)
            
            # Extract the special key
            special_key = combo[i+1:j]
            key_obj = get_special_key(special_key)
            if key_obj:
                keys.append(key_obj)
            i = j + 1
        else:
            # Regular character
            keys.append(combo[i])
            i += 1
    
    # Press all keys in the combination
    for key in keys:
        kb_controller.press(key)
    
    # Release all keys in reverse order
    for key in reversed(keys):
        kb_controller.release(key)

def press_special_key(key_name):
    """Press a special key"""
    key_obj = get_special_key(key_name)
    if key_obj:
        kb_controller.press(key_obj)
        kb_controller.release(key_obj)

def get_special_key(key_name):
    """Convert key name to pynput Key object"""
    key_map = {
        'ctrl': Key.ctrl,
        'shift': Key.shift,
        'alt': Key.alt,
        'tab': Key.tab,
        'enter': Key.enter,
        'return': Key.enter,
        'esc': Key.esc,
        'escape': Key.esc,
        'backspace': Key.backspace,
        'delete': Key.delete,
        'space': Key.space,
        'up': Key.up,
        'down': Key.down,
        'left': Key.left,
        'right': Key.right,
        'home': Key.home,
        'end': Key.end,
        'page_up': Key.page_up,
        'page_down': Key.page_down,
        'f1': Key.f1,
        'f2': Key.f2,
        'f3': Key.f3,
        'f4': Key.f4,
        'f5': Key.f5,
        'f6': Key.f6,
        'f7': Key.f7,
        'f8': Key.f8,
        'f9': Key.f9,
        'f10': Key.f10,
        'f11': Key.f11,
        'f12': Key.f12,
    }
    return key_map.get(key_name.lower())

def run_service():
    """Main function to run the service"""
    global TRIGGER_KEY, DEEPINFRA_API_KEY, SPEECH_MODEL, LLM_MODEL, SAMPLE_RATE, CHANNELS
    
    # Load environment variables
    load_dotenv()

    # Configuration from environment variables
    TRIGGER_KEY = os.getenv("TRIGGER_KEY", "&")
    DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
    SPEECH_MODEL = os.getenv("SPEECH_MODEL", "openai/whisper-large")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    CHANNELS = int(os.getenv("CHANNELS", "1"))
    
    if not DEEPINFRA_API_KEY:
        print("Error: DEEPINFRA_API_KEY not set in environment variables or .env file")
        return

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Write PID to file
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    
    print(f"Voice Command Processor running in background with PID {os.getpid()}")
    print(f"Press '{TRIGGER_KEY}' to start/stop recording.")
    print(f"To stop the service, run: kill {os.getpid()} or voice_command_stop.py")
    
    # Start key listener
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            while running:
                time.sleep(1)
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            listener.stop()
            cleanup()

def main():
    """Entry point for the script"""
    # Check if the stop command was issued
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_service()
        return
    
    # Check if the service is already running
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            # Check if process with this PID exists
            os.kill(pid, 0)
            print(f"Voice Command Processor is already running with PID {pid}")
            print(f"To stop it, run: kill {pid} or voice_command_stop.py")
            return
        except (ProcessLookupError, ValueError):
            # Process doesn't exist, remove stale PID file
            os.remove(pid_file)
        except Exception as e:
            print(f"Error checking existing process: {e}")
    
    print("Starting Voice Command Processor in background...")
    run_service()
    
    # Run as daemon
    # with daemon.DaemonContext(
    #     working_directory=os.getcwd(),
    #     stdout=open('./tmp/voice_command_processor.log', 'a'),
    #     stderr=open('./tmp/voice_command_processor.err', 'a'),
    #     detach_process=True
    # ):
    #     run_service()

def stop_service():
    """Stop the running service"""
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
    main()