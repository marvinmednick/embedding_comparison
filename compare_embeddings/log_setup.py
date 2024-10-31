import logging.config

log_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(module)s %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'embedding.log',
            'mode': 'a',
        },
    },
    'root': {
        'handlers': ['file'],
        'level': 'INFO',
    },
}


def setup_logging():
    logging.config.dictConfig(log_config)


# You can also add a function to get a logger
def get_logger(name):
    return logging.getLogger(name)
