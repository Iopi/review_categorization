import logging

# logger creating
logging.basicConfig(filename="log.txt",
                    format='%(message)s',
                    filemode='w')
logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def output(message):
    logger.info(message)
    print(message)


def exception(message):
    logger.exception(f"\nException: {message}")
    raise Exception(message)
