import logging

# logger creating
logging.basicConfig(filename="log.txt",
                    format='%(message)s',
                    filemode='w')
logging.getLogger('matplotlib.font_manager').disabled = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def output(message):
    """
    Output to console and log file
    :param message: message
    :return:
    """
    logger.info(message)
    print(message)


def exception(message):
    """
    Raise exception to log file and console
    :param message: message
    :return:
    """
    logger.exception(f"\nException: {message}")
    raise Exception(message)
