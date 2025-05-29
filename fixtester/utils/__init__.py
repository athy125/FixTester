"""
Utility modules for FixTester providing configuration management,
logging, metrics collection, helper functions, decorators, exception handling,
and other cross-cutting concerns.
"""

from .config_loader import ConfigLoader
from .logger import setup_logger, FixTesterLogger
from .metrics import MetricsCollector, PerformanceMonitor, HealthChecker, get_metrics_collector
from .helpers import (
    DataConverter, StringUtils, FileUtils, ValidationUtils, RetryUtils,
    CryptoUtils, NetworkUtils, BatchProcessor, ConfigMerger,
    generate_uuid, current_timestamp, format_duration, parse_size_string,
    chunk_list, safe_cast
)
from .decorators import (
    timer, retry, cache, validate_args, deprecated, rate_limit,
    singleton, thread_safe, log_calls, handle_exceptions, timeout,
    property_cached, memoize_method, performance_monitor,
    async_timer, async_retry, circuit_breaker, require_permissions,
    profile_memory, synchronized, validate_types, ensure_connection,
    compose, fix_message_handler
)
from .exceptions import (
    FixTesterException, FixEngineException, ConnectionException,
    SessionException, MessageException, ValidationException,
    ConfigurationException, TestException, TimeoutException,
    AuthenticationException, AuthorizationException, DataException,
    ResourceException, FixProtocolException, FixVersionException,
    SequenceException, ChecksumException, ExceptionHandler,
    ErrorCodes, exception_context, error_boundary,
    safe_execute, collect_exceptions, retry_on_exception,
    exception_to_http_status, get_exception_registry, format_exception_chain
)

__all__ = [
    # Configuration and logging
    "ConfigLoader",
    "setup_logger", 
    "FixTesterLogger",
    
    # Metrics and monitoring
    "MetricsCollector",
    "PerformanceMonitor", 
    "HealthChecker",
    "get_metrics_collector",
    
    # Helper utilities
    "DataConverter",
    "StringUtils",
    "FileUtils", 
    "ValidationUtils",
    "RetryUtils",
    "CryptoUtils",
    "NetworkUtils",
    "BatchProcessor",
    "ConfigMerger",
    "generate_uuid",
    "current_timestamp",
    "format_duration", 
    "parse_size_string",
    "chunk_list",
    "safe_cast",
    
    # Decorators
    "timer",
    "retry",
    "cache",
    "validate_args",
    "deprecated",
    "rate_limit",
    "singleton", 
    "thread_safe",
    "log_calls",
    "handle_exceptions",
    "timeout",
    "property_cached",
    "memoize_method",
    "performance_monitor",
    "async_timer",
    "async_retry", 
    "circuit_breaker",
    "require_permissions",
    "profile_memory",
    "synchronized",
    "validate_types",
    "ensure_connection",
    "compose",
    "fix_message_handler",
    
    # Exceptions
    "FixTesterException",
    "FixEngineException",
    "ConnectionException",
    "SessionException", 
    "MessageException",
    "ValidationException",
    "ConfigurationException",
    "TestException",
    "TimeoutException",
    "AuthenticationException",
    "AuthorizationException",
    "DataException",
    "ResourceException",
    "FixProtocolException", 
    "FixVersionException",
    "SequenceException",
    "ChecksumException",
    "ExceptionHandler",
    "ErrorCodes",
    "exception_context",
    "error_boundary",
    "safe_execute",
    "collect_exceptions",
    "retry_on_exception",
    "exception_to_http_status",
    "get_exception_registry",
    "format_exception_chain"
]