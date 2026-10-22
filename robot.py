# This is where your main robot code resides. It extends from the RobotInterface file.
# It inherits all the complex camera, sound, and movement functions!
# The self.command and self.routine variables are important because they keep track of robot states. 
# Remember: Flask uses Threading (multiple processes running at once, which can confuse the robot if states aren't tracked)

import logging, sys, os, time
from interfaces.robotinterface import RobotInterface
from interfaces.databaseinterface import Database

class Robot(RobotInterface): 
    
    def __init__(self, DATABASE=None):
        # Initialize the underlying hardware and camera threads
        super().__init__()
        
        # Connect the database securely
        if DATABASE is None:
            self.DATABASE = Database("databases/test.db")
        else:
            self.DATABASE = DATABASE
            
        # State tracking for Flask integration
        self.routine = "ready" 
        return
     
    # Write a function for automated search
    def automated_search(self, timelimit=300):
        """
        Students can write their autonomous logic here!
        Use self.move_direction_until_detection(), self.pick_up_centered_object_with_look_down(), etc.
        """
        self.routine = 'automated_search'
        self.logger.info('Beginning Automated Search')
        
        # STUDENT CODE GOES HERE
        
        return
    
    def stop_automated_search(self):
        """Safely interrupts the automated routine."""
        self.routine = 'ready'
        self.logger.info('Stop Automated Search')
        self.stop_command() # Use the inherited failsafe stop
        return
    
# Only execute if this is the main file, good for testing code
if __name__ == '__main__':
    print("\033c")
    ROBOT = Robot(DATABASE=None)
    
    try:
        ROBOT.stop()
        input("Press enter to begin testing:")
        
        # You can test student functions here
        # ROBOT.automated_search(timelimit=10)
        
    except KeyboardInterrupt:
        print("\n[!] Testing interrupted.")
    finally:
        # Guarantee safe shutdown
        ROBOT.shutdown()