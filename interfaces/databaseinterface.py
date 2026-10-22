#----------------------------------------------------------------------------
# This Database class provides an interface to the database
# It's therefore easier to simply inherit the code..
# Created by Brad Nielsen 2019
#-----------------------------------------------------------------------#
import sqlite3
import logging
from contextlib import closing

class Database:

    def __init__(self, location="", log=None):
        """Initializes the database connection parameters."""
        self.location = location
        # Safely default to the module logger if none is passed
        self.logger = log or logging.getLogger(__name__)

    def connect(self):
        """Returns a configured handle to the Database connection."""
        connection = sqlite3.connect(self.location)
        connection.row_factory = sqlite3.Row 
        
        # Enable WAL mode for simultaneous reads/writes
        connection.execute("PRAGMA journal_mode=WAL;") 
        
        return connection

    def ViewQuery(self, query, params=()):
        """
        Executes a SELECT query and returns the results.
        Returns an empty list [] if no results are found or if an error occurs.
        """
        result = []
        try:
            # closing() guarantees connection.close() is called when the block ends
            with closing(self.connect()) as connection:
                cursor = connection.execute(query, params)
                records = cursor.fetchall()
                if records:
                    result = [dict(row) for row in records]
                    
        except sqlite3.Error as e:
            self.logger.error(f"DATABASE ERROR: {e}")
            self.logger.error(f"QUERY: {query}")
            
        return result

    def ModifyQuery(self, query, params=()):
        """
        Executes an INSERT, UPDATE, or DELETE query.
        Returns True on success, False on failure.
        """
        success = False
        try:
            with closing(self.connect()) as connection:
                # 'with connection:' automatically handles .commit() on success 
                # and .rollback() if an exception is thrown
                with connection:
                    connection.execute(query, params)
                success = True
                
        except sqlite3.Error as e:
            self.logger.error(f"DATABASE ERROR: {e}")
            self.logger.error(f"QUERY: {query}")
            
        return success

    def log(self, message):
        self.logger.info(message)

    def log_error(self, error):
        self.logger.error(error)