#!/usr/bin/env python3
"""
Rapid Photo Capture for Emotion Dataset
Hold spacebar to capture 100+ photos at MAX FPS for emotion_project dataset
Optimized for Jetson Orin Nano
"""

import cv2
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

# Load emotion labels
EMOTION_PROJECT = Path.home() / "emotion_project"
LABELS_FILE = EMOTION_PROJECT / "labels.txt"

try:
    with open(LABELS_FILE, "r") as f:
        EMOTIONS = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    sys.exit(1)

def select_emotion():
    """Let user select which emotion to capture photos for"""
    print("\n" + "="*50)
    print("📊 SELECT EMOTION CATEGORY")
    print("="*50)
    for i, emotion in enumerate(EMOTIONS, 1):
        print(f"{i}. {emotion.upper()}")
    
    while True:
        try:
            choice = int(input(f"\nSelect emotion (1-{len(EMOTIONS)}): "))
            if 1 <= choice <= len(EMOTIONS):
                return EMOTIONS[choice - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(EMOTIONS)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def create_output_dir(emotion):
    """Create output directory for the selected emotion"""
    output_dir = EMOTION_PROJECT / "dataset" / "train" / emotion / "raw_capture"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def main():
    global cap, emotion
    
    # Select emotion
    emotion = select_emotion()
    output_dir = create_output_dir(emotion)
    
    print(f"\n✅ Capturing photos for: {emotion.upper()}")
    print(f"📁 Saving to: {output_dir}\n")
    
    # Initialize camera
    print("🎥 Initializing USB camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera. Check USB camera connection.")
        return 1
    
    try:
        # Optimize for maximum FPS - use simpler settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to read initial frame to verify camera works
        ret, _ = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            cap.release()
            return 1
        
        print("✅ Camera ready!")
        print("\n" + "="*60)
        print("📹 INSTRUCTIONS:")
        print("  • HOLD SPACEBAR to capture photos (as fast as possible)")
        print("  • Release SPACEBAR to stop capturing")
        print("  • Press 'Q' to quit")
        print("="*60 + "\n")
        
        capturing = False
        photo_count = 0
        capture_start_time = None
        last_spacebar_state = False  # Track spacebar state from last iteration
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Minimal processing to avoid crashes
            display_frame = frame.copy()
            
            # Add simple status text
            status = f"EMOTION: {emotion.upper()}"
            if capturing:
                elapsed = time.time() - capture_start_time
                fps = photo_count / elapsed if elapsed > 0 else 0
                status += f" | CAPTURING: {photo_count} @ {fps:.1f} fps"
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                status += " | HOLD SPACEBAR TO START"
                cv2.putText(display_frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            cv2.putText(display_frame, "Press Q to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display with error handling
            try:
                cv2.imshow("Capture", display_frame)
            except Exception as e:
                print(f"⚠️  Display error: {e}")
            
            # Keyboard input - check immediately
            key = cv2.waitKey(1) & 0xFF
            
            # Detect spacebar press/release
            spacebar_pressed = (key == 32)  # 32 = spacebar
            
            # Start capturing when spacebar is first pressed
            if spacebar_pressed and not last_spacebar_state:
                capturing = True
                photo_count = 0
                capture_start_time = time.time()
                print(f"🔴 CAPTURING STARTED for {emotion.upper()}...")
            
            # Stop capturing when spacebar is released
            if not spacebar_pressed and last_spacebar_state and capturing:
                elapsed = time.time() - capture_start_time
                fps = photo_count / elapsed if elapsed > 0 else 0
                print(f"\n✅ CAPTURE STOPPED!")
                print(f"  📊 Total photos: {photo_count}")
                print(f"  ⏱️  Time: {elapsed:.2f} seconds")
                print(f"  📈 Capture rate: {fps:.1f} fps")
                print(f"  📁 Saved to: {output_dir}\n")
                capturing = False
            
            last_spacebar_state = spacebar_pressed
            
            # Q to quit
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Exiting...")
                break
            
            # Capture frame if capturing
            if capturing:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                    filename = f"{timestamp}{photo_count:05d}.jpg"
                    filepath = output_dir / filename
                    
                    # Write with compression to reduce memory
                    cv2.imwrite(str(filepath), frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                    photo_count += 1
                    
                    if photo_count % 20 == 0:
                        print(f"  ✓ {photo_count} photos...")
                except Exception as e:
                    print(f"❌ Error saving photo: {e}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        print("\n" + "="*60)
        print(f"✅ Session complete!")
        print("="*60)

if __name__ == "__main__":
    main()
