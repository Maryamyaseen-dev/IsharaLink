import os
import asyncio
import edge_tts
import pygame
import threading
import hashlib

# ==========================================
# ISHARALINK - URDU TEXT-TO-SPEECH (TTS)
# ==========================================

VOICE = "ur-PK-UzmaNeural"  # Fluent Pakistani Urdu Voice
CACHE_DIR = "tts_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize Pygame Mixer for non-blocking audio playback
try:
    pygame.mixer.init()
except Exception as e:
    print(f"[Warning] Audio mixer init: {e}")

def _generate_and_play(text):
    """Worker function to generate and play audio in background thread"""
    try:
        # Create unique filename based on text hash
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        file_path = os.path.join(CACHE_DIR, f"{text_hash}.mp3")

        # If not cached, generate using Edge-TTS
        if not os.path.exists(file_path):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            communicate = edge_tts.Communicate(text, VOICE)
            loop.run_until_complete(communicate.save(file_path))
            loop.close()

        # Play audio via Pygame
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
    except Exception as err:
        print(f"[TTS Error] {err}")

def speak_urdu(text):
    """
    Speaks the given Urdu text in a background thread.
    Zero lag on camera/UI loop!
    """
    if not text or not text.strip():
        return
    thread = threading.Thread(target=_generate_and_play, args=(text.strip(),), daemon=True)
    thread.start()

if __name__ == "__main__":
    print("Testing Urdu Speaker...")
    speak_urdu("السلام علیکم! اشارہ لنک سسٹم تیار ہے۔")
    import time
    time.sleep(3)
    print("Test finished.")
