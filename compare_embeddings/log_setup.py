import logging.config

LOGGER_BASE_NAME = 'embed'
# Define the new TRACE level
TRACE_LEVEL_NUM = 5  # This should be less than DEBUG (10)


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)


# Add the trace method to the Logger class
logging.Logger.trace = trace

# Set up basic configuration for logging
# logging.basicConfig(level=TRACE_LEVEL_NUM,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.basicConfig(level=TRACE_LEVEL_NUM, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# You can also add a function to get a logger
def get_logger(name):
    return logging.getLogger(f"{LOGGER_BASE_NAME}.{name}")


def update_log_levels(new_level):
    """Update the log level for all custom loggers."""
    for name in logging.root.manager.loggerDict:
        if name == LOGGER_BASE_NAME or name.startswith(f"{LOGGER_BASE_NAME}."):
            logging.getLogger(name).setLevel(new_level)


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


