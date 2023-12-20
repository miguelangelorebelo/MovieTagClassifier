import logging
import sys

from loguru import logger

from configs.logger_config import logger_config


class CustomLogger:
    @classmethod
    def make_logger(cls):
        logger.remove()  # comment to see in console
        logger.add(sys.stdout, level=logger_config.get("LOG_LEVEL"), enqueue=True, backtrace=True)
        logger.add(
            logger_config.get("LOG_PATH") + "/{time}.log",
            rotation=logger_config.get("LOG_ROTATION"),
            level=logger_config.get("LOG_LEVEL"),
            compression=logger_config.get("LOG_COMPRESSION"),
            enqueue=True,
            backtrace=False
        )
        logging.basicConfig(handlers=[cls.InterceptHandler()], level=0)
        logging.getLogger("uvicorn.access").handlers = [cls.InterceptHandler()]
        for _log in ['uvicorn', 'uvicorn.error', 'fastapi']:
            _logger = logging.getLogger(_log)
            _logger.handlers = [cls.InterceptHandler()]

        return logger.bind(request_id=None, method=None)

    class InterceptHandler(logging.Handler):  # based on official doc, bind log to loguru
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = self.logLevel.get(record.levelno)

            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            _log = logger.bind(request_id='app')
            _log.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
