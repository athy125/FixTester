"""
Logging utilities for FixTester with structured logging and multiple output formats.
"""
import os
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """Setup a logger with the specified configuration.

    Args:
        name: Logger name
        config: Logging configuration dictionary
        level: Log level override

    Returns:
        Configured logger instance
    """
    if config is None:
        config = {}
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set log level
    log_level = level or config.get('level', 'INFO')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    console_formatter = _create_console_formatter(config)
    file_formatter = _create_file_formatter(config)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Setup file handler if specified
    log_file = config.get('file')
    if log_file:
        file_handler = _create_file_handler(log_file, config)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent log messages from being handled by root logger
    logger.propagate = False
    
    return logger


def _create_console_formatter(config: Dict[str, Any]) -> logging.Formatter:
    """Create a console log formatter.

    Args:
        config: Logging configuration

    Returns:
        Console formatter
    """
    console_format = config.get('console_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    # Add colors if supported
    if config.get('use_colors', True) and _supports_color():
        console_format = _add_colors_to_format(console_format)
    
    return logging.Formatter(console_format, datefmt=date_format)


def _create_file_formatter(config: Dict[str, Any]) -> logging.Formatter:
    """Create a file log formatter.

    Args:
        config: Logging configuration

    Returns:
        File formatter
    """
    file_format = config.get(
        'file_format',
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    date_format = config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    return logging.Formatter(file_format, datefmt=date_format)


def _create_file_handler(log_file: str, config: Dict[str, Any]) -> logging.Handler:
    """Create a file log handler.

    Args:
        log_file: Path to the log file
        config: Logging configuration

    Returns:
        File handler
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if rotation is enabled
    max_bytes = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB default
    backup_count = config.get('backup_count', 5)
    
    if max_bytes > 0:
        # Use rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        # Use regular file handler
        handler = logging.FileHandler(log_file, encoding='utf-8')
    
    return handler


def _supports_color() -> bool:
    """Check if the terminal supports colors.

    Returns:
        True if colors are supported, False otherwise
    """
    return (
        hasattr(sys.stdout, 'isatty') and 
        sys.stdout.isatty() and
        'TERM' in os.environ and
        os.environ['TERM'] != 'dumb'
    )


def _add_colors_to_format(format_string: str) -> str:
    """Add color codes to log format string.

    Args:
        format_string: Original format string

    Returns:
        Format string with color codes
    """
    # ANSI color codes
    colors = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Replace levelname with colored version
    colored_format = format_string.replace(
        '%(levelname)s',
        f'{colors["%(levelname)s"]}%(levelname)s{colors["RESET"]}'
    )
    
    # This is a simplified approach - a more sophisticated implementation
    # would use a custom formatter to apply colors based on the actual log level
    return colored_format


class FixTesterLogger:
    """Enhanced logger for FixTester with structured logging capabilities."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the FixTester logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.config = config or {}
        self.logger = setup_logger(name, self.config)
        self.name = name
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message with optional structured data.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message with optional structured data.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message with optional structured data.
        
        Args:
            message: Log message
            **kwargs: Additional structured data
        """
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message with optional exception and structured data.
        
        Args:
            message: Log message
            exception: Optional exception to log
            **kwargs: Additional structured data
        """
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['exception_type'] = type(exception).__name__
        
        self._log(logging.ERROR, message, **kwargs)
        
        if exception:
            self.logger.exception("Exception details:")
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log a critical message with optional exception and structured data.
        
        Args:
            message: Log message
            exception: Optional exception to log
            **kwargs: Additional structured data
        """
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['exception_type'] = type(exception).__name__
        
        self._log(logging.CRITICAL, message, **kwargs)
        
        if exception:
            self.logger.exception("Exception details:")
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal method to log with structured data.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Structured data
        """
        if kwargs:
            # Format structured data
            structured_data = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            full_message = f"{message} | {structured_data}"
        else:
            full_message = message
        
        self.logger.log(level, full_message)
    
    def log_fix_message(self, direction: str, message: str, session_id: Optional[str] = None) -> None:
        """Log a FIX message with structured information.
        
        Args:
            direction: Message direction ('SENT' or 'RECEIVED')
            message: FIX message string
            session_id: Optional session identifier
        """
        self.info(
            f"FIX Message {direction}",
            session_id=session_id,
            message_length=len(message),
            message=message[:200] + "..." if len(message) > 200 else message
        )
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            **kwargs: Additional metrics
        """
        self.info(
            f"Performance: {operation}",
            duration_seconds=duration,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_test_result(self, test_name: str, passed: bool, duration: float, error: Optional[str] = None) -> None:
        """Log test execution results.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            duration: Test duration in seconds
            error: Optional error message
        """
        level = logging.INFO if passed else logging.ERROR
        
        self._log(
            level,
            f"Test {'PASSED' if passed else 'FAILED'}: {test_name}",
            test_name=test_name,
            passed=passed,
            duration_seconds=duration,
            error=error
        )

