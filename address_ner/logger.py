import logging


def get_logger(name=None):
    if name is None:
        log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    else:
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(name or "")
    return logger
