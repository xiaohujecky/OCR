import logging
from logging.handlers import RotatingFileHandler


# init logger here
class ApiHandler(object):
    def __init__(self):
        pass

    @property
    def logger(self, log_file='img_server.log', log_level=logging.INFO, output_level=logging.INFO):
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %a %H:%M:%S',
                            filename='%s' % log_file)
        logger = logging.getLogger("img_server")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        rt = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
        rt.setLevel(output_level)
        rt.setFormatter(formatter)

        logger.addHandler(rt)
        logger.removeHandler(rt)
        return logger

api = ApiHandler()
