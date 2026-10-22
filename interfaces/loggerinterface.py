import logging, os

def setup_logger(logger, filename):
    logger.setLevel(logging.DEBUG)
    filetemp = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    file_handler = logging.FileHandler(filetemp)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return