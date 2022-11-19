import logging

# now we will Create and configure logger
logging.basicConfig(filename="std.log",
                    format='%(message)s',
                    filemode='w')
# logging.basicConfig(filename="std.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')

logging.getLogger('matplotlib.font_manager').disabled = True

# Let us Create an object
logger = logging.getLogger()

# Now we are going to Set the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def output(message):
    logger.info(message)
    print(message)
