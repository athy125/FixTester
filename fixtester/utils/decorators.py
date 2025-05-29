"""
Decorators for FixTester providing timing, caching, validation, retry logic,
and other cross-cutting concerns.
"""
import time
import functools
import threading
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from collections import OrderedDict
import warnings

from fixtester.utils.logger import setup_logger
from fixtester.utils.metrics import get_metrics_collector

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def timer(metric_name: Optional[str] = None, log_result: bool = True):
    """Decorator to time function execution.

    Args:
        metric_name: Name for the metric (defaults to function name)
        log_result: Whether to log the timing result

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                metrics = get_metrics_collector()
                name = metric_name or f"{func.__module__}.{func.__name__}"
                metrics.record_timer(name, duration, {
                    'function': func.__name__,
                    'success': str(success).lower()
                })
                
                # Log result if requested
                if log_result:
                    logger = setup_logger(func.__module__)
                    if success:
                        logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                    else:
                        logger.warning(f"{func.__name__} failed after {duration:.3f}s: {error}")
            
            return result
        
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None
):
    """Decorator to retry function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on
        on_failure: Optional callback to execute on each failure

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Call failure callback if provided
                    if on_failure:
                        try:
                            on_failure(e, attempt + 1, max_attempts)
                        except Exception:
                            pass  # Don't let callback errors affect retry logic
                    
                    # Don't delay after the last attempt
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # All attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def cache(max_size: int = 128, ttl: Optional[float] = None):
    """Decorator to cache function results with optional TTL.

    Args:
        max_size: Maximum number of cached results
        ttl: Time to live in seconds (None for no expiration)

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        cache_dict = OrderedDict()
        cache_lock = threading.RLock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = _make_cache_key(args, kwargs)
            current_time = time.time()
            
            with cache_lock:
                # Check if result is cached and not expired
                if key in cache_dict:
                    result, timestamp = cache_dict[key]
                    
                    if ttl is None or current_time - timestamp < ttl:
                        # Move to end (LRU)
                        cache_dict.move_to_end(key)
                        return result
                    else:
                        # Expired, remove from cache
                        del cache_dict[key]
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache_dict[key] = (result, current_time)
                
                # Enforce max size
                if len(cache_dict) > max_size:
                    cache_dict.popitem(last=False)  # Remove oldest item
                
                return result
        
        # Add cache management methods
        def clear_cache():
            with cache_lock:
                cache_dict.clear()
        
        def cache_info():
            with cache_lock:
                return {
                    'size': len(cache_dict),
                    'max_size': max_size,
                    'ttl': ttl
                }
        
        wrapper.clear_cache = clear_cache
        wrapper.cache_info = cache_info
        
        return wrapper
    return decorator


def validate_args(*validators):
    """Decorator to validate function arguments.

    Args:
        *validators: Validation functions that take an argument and return bool

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for i, (arg, validator) in enumerate(zip(args, validators)):
                if not validator(arg):
                    raise ValueError(f"Validation failed for argument {i} in {func.__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def deprecated(reason: str = "This function is deprecated", version: Optional[str] = None):
    """Decorator to mark functions as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version when deprecated

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            message += f": {reason}"
            
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(max_calls: int, period: float):
    """Decorator to rate limit function calls.

    Args:
        max_calls: Maximum number of calls allowed
        period: Time period in seconds

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        calls = []
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            with lock:
                # Remove old calls outside the time window
                calls[:] = [call_time for call_time in calls if current_time - call_time < period]
                
                # Check if we've exceeded the rate limit
                if len(calls) >= max_calls:
                    oldest_call = min(calls)
                    sleep_time = period - (current_time - oldest_call)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        current_time = time.time()
                        # Remove old calls again after sleeping
                        calls[:] = [call_time for call_time in calls if current_time - call_time < period]
                
                # Record this call
                calls.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls):
    """Decorator to make a class a singleton.

    Args:
        cls: Class to make singleton

    Returns:
        Singleton class
    """
    instances = {}
    lock = threading.Lock()
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def thread_safe(func: F) -> F:
    """Decorator to make a function thread-safe using a lock.

    Args:
        func: Function to make thread-safe

    Returns:
        Thread-safe function
    """
    lock = threading.RLock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    
    return wrapper


def log_calls(logger_name: Optional[str] = None, log_args: bool = False, log_result: bool = False):
    """Decorator to log function calls.

    Args:
        logger_name: Name of logger to use (defaults to function module)
        log_args: Whether to log function arguments
        log_result: Whether to log function result

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        logger = setup_logger(logger_name or func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log function entry
            msg = f"Calling {func.__name__}"
            if log_args and (args or kwargs):
                msg += f" with args={args}, kwargs={kwargs}"
            logger.debug(msg)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                msg = f"Completed {func.__name__}"
                if log_result:
                    msg += f" with result={result}"
                logger.debug(msg)
                
                return result
                
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator


def handle_exceptions(*exception_types, default_return=None, log_errors=True):
    """Decorator to handle specific exceptions.

    Args:
        *exception_types: Exception types to handle
        default_return: Default value to return on exception
        log_errors: Whether to log caught exceptions

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_errors:
                    logger = setup_logger(func.__module__)
                    logger.error(f"Handled exception in {func.__name__}: {e}")
                
                return default_return
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """Decorator to add timeout to function execution.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up signal handler (Unix only)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Windows fallback - use threading
                import threading
                
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(seconds)
                
                if thread.is_alive():
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
        
        return wrapper
    return decorator


def property_cached(func: F) -> F:
    """Decorator to cache property values.

    Args:
        func: Property getter function

    Returns:
        Cached property
    """
    attr_name = f'_cached_{func.__name__}'
    
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return property(wrapper)


def memoize_method(func: F) -> F:
    """Decorator to memoize instance method results.

    Args:
        func: Method to memoize

    Returns:
        Memoized method
    """
    cache_attr = f'_memoize_cache_{func.__name__}'
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        
        cache = getattr(self, cache_attr)
        key = _make_cache_key(args, kwargs)
        
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
        
        return cache[key]
    
    return wrapper


def _make_cache_key(args: tuple, kwargs: dict) -> str:
    """Create a cache key from function arguments.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    import hashlib
    import json
    
    # Convert arguments to a hashable representation
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    key_str = json.dumps(key_data, default=str, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


# Context manager decorators
class performance_monitor:
    """Context manager for monitoring function performance."""
    
    def __init__(self, name: str, log_result: bool = True):
        """Initialize performance monitor.
        
        Args:
            name: Name for the operation being monitored
            log_result: Whether to log the performance result
        """
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.logger = setup_logger(__name__)
        
    def __enter__(self):
        """Enter the performance monitoring context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the performance monitoring context."""
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Record metrics
            metrics = get_metrics_collector()
            metrics.record_timer(f"performance_{self.name}", duration, {
                'operation': self.name,
                'success': str(exc_type is None).lower()
            })
            
            # Log result if requested
            if self.log_result:
                if exc_type is None:
                    self.logger.info(f"Operation '{self.name}' completed in {duration:.3f}s")
                else:
                    self.logger.warning(f"Operation '{self.name}' failed after {duration:.3f}s")


def async_timer(metric_name: Optional[str] = None, log_result: bool = True):
    """Decorator to time async function execution.

    Args:
        metric_name: Name for the metric (defaults to function name)
        log_result: Whether to log the timing result

    Returns:
        Decorated async function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                metrics = get_metrics_collector()
                name = metric_name or f"{func.__module__}.{func.__name__}"
                metrics.record_timer(name, duration, {
                    'function': func.__name__,
                    'success': str(success).lower(),
                    'async': 'true'
                })
                
                # Log result if requested
                if log_result:
                    logger = setup_logger(func.__module__)
                    if success:
                        logger.debug(f"Async {func.__name__} completed in {duration:.3f}s")
                    else:
                        logger.warning(f"Async {func.__name__} failed after {duration:.3f}s: {error}")
            
            return result
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to retry async function on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated async function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Don't delay after the last attempt
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # All attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


class circuit_breaker:
    """Circuit breaker decorator to prevent cascading failures."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit break
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # Circuit state
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
        
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                # Check if circuit should be closed
                if self.state == 'OPEN':
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = 'HALF_OPEN'
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - reset failure count
                    if self.state == 'HALF_OPEN':
                        self.state = 'CLOSED'
                    self.failure_count = 0
                    
                    return result
                    
                except self.expected_exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    # Open circuit if threshold exceeded
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'OPEN'
                    
                    raise
        
        return wrapper


def require_permissions(*required_perms):
    """Decorator to check permissions before function execution.

    Args:
        *required_perms: Required permissions

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This is a placeholder - in a real implementation,
            # you would check user permissions here
            
            # For now, just log the permission check
            logger = setup_logger(func.__module__)
            logger.debug(f"Permission check for {func.__name__}: {required_perms}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def profile_memory(func: F) -> F:
    """Decorator to profile memory usage of a function.

    Args:
        func: Function to profile

    Returns:
        Decorated function with memory profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss
            mem_diff = mem_after - mem_before
            
            # Record memory usage metrics
            metrics = get_metrics_collector()
            metrics.record_histogram(f"memory_usage_{func.__name__}", mem_diff, {
                'function': func.__name__,
                'module': func.__module__
            })
            
            logger = setup_logger(func.__module__)
            logger.debug(f"Memory usage for {func.__name__}: {mem_diff / 1024 / 1024:.2f} MB")
            
            return result
            
        except ImportError:
            # psutil not available, just run the function
            return func(*args, **kwargs)
    
    return wrapper


def synchronized(lock_name: str = None):
    """Decorator to synchronize method calls using a named lock.

    Args:
        lock_name: Name of the lock attribute (defaults to '_sync_lock')

    Returns:
        Decorated method
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            lock_attr = lock_name or '_sync_lock'
            
            # Create lock if it doesn't exist
            if not hasattr(self, lock_attr):
                setattr(self, lock_attr, threading.RLock())
            
            lock = getattr(self, lock_attr)
            
            with lock:
                return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def validate_types(**type_checks):
    """Decorator to validate argument types.

    Args:
        **type_checks: Mapping of argument names to expected types

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Check types
            for arg_name, expected_type in type_checks.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{arg_name}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def ensure_connection(connection_check_method: str = 'is_connected'):
    """Decorator to ensure connection before method execution.

    Args:
        connection_check_method: Name of method to check connection status

    Returns:
        Decorated method
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if connection check method exists
            if hasattr(self, connection_check_method):
                check_method = getattr(self, connection_check_method)
                if not check_method():
                    raise ConnectionError(f"Not connected - cannot execute {func.__name__}")
            
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


# Utility function to combine multiple decorators
def compose(*decorators):
    """Compose multiple decorators into a single decorator.

    Args:
        *decorators: Decorators to compose

    Returns:
        Composed decorator
    """
    def decorator(func):
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return decorator


# Example usage combinations
def fix_message_handler(
    metric_name: Optional[str] = None,
    max_attempts: int = 3,
    timeout_seconds: Optional[float] = None
):
    """Combined decorator for FIX message handling functions.

    Args:
        metric_name: Name for timing metrics
        max_attempts: Maximum retry attempts
        timeout_seconds: Optional timeout

    Returns:
        Combined decorator
    """
    decorators = [
        timer(metric_name),
        retry(max_attempts=max_attempts),
        log_calls(log_args=True)
    ]
    
    if timeout_seconds:
        decorators.append(timeout(timeout_seconds))
    
    return compose(*decorators)