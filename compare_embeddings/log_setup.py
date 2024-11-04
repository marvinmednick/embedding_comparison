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
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
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


def switch_to_handler(handler_name):
    root_logger = logging.getLogger()
    
    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the specified handler
    if handler_name in log_config['handlers']:
        handler_config = log_config['handlers'][handler_name]
        if handler_config['class'] == 'logging.FileHandler':
            handler = logging.FileHandler(handler_config['filename'])
        elif handler_config['class'] == 'logging.StreamHandler':
            handler = logging.StreamHandler()
        
        handler.setFormatter(logging.Formatter(log_config['formatters']['standard']['format']))
        handler.setLevel(handler_config['level'])
        root_logger.addHandler(handler)
    else:
        raise ValueError(f"Unknown handler: {handler_name}")
