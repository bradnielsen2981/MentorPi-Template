#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, threading
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pygame
import speake3
from loggerinterface import setup_logger
import logging

class SoundInterface():
    
    def __init__(self):
        """Initializes the Text-to-Speech engine and Pygame music mixer."""
        self.engine = speake3.Speake()
        self.engine.set('voice', 'en-scotish')
        self.engine.set('speed', '150')
        self.engine.set('pitch', '60')
        
        pygame.mixer.init() # load music player
        
        self.logger = logging.getLogger('SoundInterface')
        setup_logger(self.logger, '../logs/sound.log')
        
        # A lock to prevent the robot from trying to say two things at the exact same time
        self.speak_lock = threading.Lock()
    
    def get_all_voices(self):
        """Prints all available voices to the console."""
        voices = self.engine.get("voices") # shows all the voices that could be selected
        for voice in voices:
            print(voice)
            
        voices_2 = self.engine.get("voices", "en") # shows all english voices
        for voice in voices_2:
            print(voice)
    
    def say(self, message):
        """
        Gets the robot to speak a phrase out loud.
        Uses a background thread so the robot doesn't freeze while talking!
        """
        def _speak_thread():
            # The lock ensures that if say() is called twice quickly, 
            # it finishes the first sentence before starting the next.
            with self.speak_lock:
                self.engine.say(message) 
                self.engine.talkback()

        # Fire and forget the speech in the background
        thread = threading.Thread(target=_speak_thread, daemon=True)
        thread.start()

    def load_mp3(self, song): 
        """Loads an MP3 file into the Pygame mixer."""
        pygame.mixer.music.load(song)
    
    def play_music(self, times=-1):
        """Plays the loaded MP3. Default is infinite loop (-1)."""
        pygame.mixer.music.play(times)
    
    def pause_music(self):
        """Pauses the currently playing MP3."""
        pygame.mixer.music.pause()
    
    def unpause_music(self):
        """Resumes a paused MP3."""
        pygame.mixer.music.unpause()

    def stop_music(self):
        """Completely stops the MP3 playback."""
        pygame.mixer.music.stop()

    def set_volume(self, v=0.8):
        """Sets the volume of the Pygame mixer (0.0 to 1.0)."""
        pygame.mixer.music.set_volume(v)

#---------------------------------------------
# only execute if this is the main file, good for testing code.   
if __name__ == "__main__":
    SOUND = SoundInterface()
    print("\033c")
    print("HERE")
    
    # Use a try/except block just in case the music file is missing on your testing PC
    try:
        SOUND.load_mp3("static/music/missionimpossible.mp3")
    except pygame.error as e:
        print(f"Could not load music file: {e}")
        
    input("Press Enter to start:")
    
    # This will now speak AND play music at the exact same time without freezing!
    SOUND.say("Hello, my name is WALLEE")
    
    try:
        SOUND.play_music(1)
    except pygame.error:
        pass
        
    response = input("Press Enter to stop")
    SOUND.stop_music()