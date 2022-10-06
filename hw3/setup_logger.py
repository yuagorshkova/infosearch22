import logging

logger = logging.getLogger('infosearch_logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())