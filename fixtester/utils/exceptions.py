"""
Custom exceptions for FixTester providing specific error types,
error context, and structured error handling.
"""
from typing import Any, Dict, List, Optional, Union
import traceback


class FixTesterException(Exception):
    """Base exception class for all FixTester errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize FixTester exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            context: Optional context information
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.timestamp = None
        
        # Capture stack trace
        self.stack_trace = traceback.format_stack()
        
    def __str__(self) -> str:
        """Return string representation of the exception."""
        result = self.message
        
        if self.error_code:
            result = f"[{self.error_code}] {result}"
            
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            result += f" (Context: {context_str})"
            
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation.
        
        Returns:
            Dictionary containing exception details
        """
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'cause': str(self.cause) if self.cause else None,
            'timestamp': self.timestamp
        }


class FixEngineException(FixTesterException):
    """Exception raised by the FIX engine."""
    pass


class ConnectionException(FixEngineException):
    """Exception raised for connection-related errors."""
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        """Initialize connection exception.
        
        Args:
            message: Error message
            host: Host that failed to connect
            port: Port that failed to connect
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if host:
            context['host'] = host
        if port:
            context['port'] = port
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port


class SessionException(FixEngineException):
    """Exception raised for FIX session-related errors."""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        sender_comp_id: Optional[str] = None,
        target_comp_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize session exception.
        
        Args:
            message: Error message
            session_id: Session identifier
            sender_comp_id: Sender CompID
            target_comp_id: Target CompID
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if session_id:
            context['session_id'] = session_id
        if sender_comp_id:
            context['sender_comp_id'] = sender_comp_id
        if target_comp_id:
            context['target_comp_id'] = target_comp_id
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.session_id = session_id
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id


class MessageException(FixTesterException):
    """Exception raised for FIX message-related errors."""
    
    def __init__(
        self,
        message: str,
        message_type: Optional[str] = None,
        cl_ord_id: Optional[str] = None,
        fix_message: Optional[str] = None,
        **kwargs
    ):
        """Initialize message exception.
        
        Args:
            message: Error message
            message_type: FIX message type
            cl_ord_id: Client order ID
            fix_message: Raw FIX message string
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if message_type:
            context['message_type'] = message_type
        if cl_ord_id:
            context['cl_ord_id'] = cl_ord_id
        if fix_message:
            # Truncate long messages for context
            context['fix_message'] = fix_message[:200] + "..." if len(fix_message) > 200 else fix_message
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.message_type = message_type
        self.cl_ord_id = cl_ord_id
        self.fix_message = fix_message


class ValidationException(FixTesterException):
    """Exception raised for message validation errors."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[str] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        """Initialize validation exception.
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            field_value: Value of the field that failed validation
            validation_rule: Name of the validation rule that failed
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if field_value:
            context['field_value'] = field_value
        if validation_rule:
            context['validation_rule'] = validation_rule
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class ConfigurationException(FixTesterException):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize configuration exception.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_file: Configuration file that caused the error
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_file = config_file


class TestException(FixTesterException):
    """Exception raised during test execution."""
    
    def __init__(
        self,
        message: str,
        test_name: Optional[str] = None,
        scenario_name: Optional[str] = None,
        test_step: Optional[str] = None,
        **kwargs
    ):
        """Initialize test exception.
        
        Args:
            message: Error message
            test_name: Name of the test that failed
            scenario_name: Name of the scenario
            test_step: Current test step
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if test_name:
            context['test_name'] = test_name
        if scenario_name:
            context['scenario_name'] = scenario_name
        if test_step:
            context['test_step'] = test_step
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.test_name = test_name
        self.scenario_name = scenario_name
        self.test_step = test_step


class TimeoutException(FixTesterException):
    """Exception raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """Initialize timeout exception.
        
        Args:
            message: Error message
            timeout_seconds: Timeout duration that was exceeded
            operation: Operation that timed out
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        if operation:
            context['operation'] = operation
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class AuthenticationException(FixTesterException):
    """Exception raised for authentication failures."""
    
    def __init__(
        self,
        message: str,
        username: Optional[str] = None,
        **kwargs
    ):
        """Initialize authentication exception.
        
        Args:
            message: Error message
            username: Username that failed authentication
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if username:
            context['username'] = username
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.username = username


class AuthorizationException(FixTesterException):
    """Exception raised for authorization failures."""
    
    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        user_permissions: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize authorization exception.
        
        Args:
            message: Error message
            required_permission: Permission that was required
            user_permissions: Permissions the user has
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if required_permission:
            context['required_permission'] = required_permission
        if user_permissions:
            context['user_permissions'] = user_permissions
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.required_permission = required_permission
        self.user_permissions = user_permissions


class DataException(FixTesterException):
    """Exception raised for data-related errors."""
    
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        data_source: Optional[str] = None,
        **kwargs
    ):
        """Initialize data exception.
        
        Args:
            message: Error message
            data_type: Type of data that caused the error
            data_source: Source of the data
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if data_type:
            context['data_type'] = data_type
        if data_source:
            context['data_source'] = data_source
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.data_source = data_source


class ResourceException(FixTesterException):
    """Exception raised for resource-related errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize resource exception.
        
        Args:
            message: Error message
            resource_type: Type of resource
            resource_id: Resource identifier
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        if resource_id:
            context['resource_id'] = resource_id
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Specific FIX protocol exceptions
class FixProtocolException(MessageException):
    """Exception for FIX protocol violations."""
    pass


class FixVersionException(FixProtocolException):
    """Exception for FIX version mismatches."""
    
    def __init__(
        self,
        message: str,
        expected_version: Optional[str] = None,
        actual_version: Optional[str] = None,
        **kwargs
    ):
        """Initialize FIX version exception.
        
        Args:
            message: Error message
            expected_version: Expected FIX version
            actual_version: Actual FIX version
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if expected_version:
            context['expected_version'] = expected_version
        if actual_version:
            context['actual_version'] = actual_version
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.expected_version = expected_version
        self.actual_version = actual_version


class SequenceException(FixProtocolException):
    """Exception for sequence number errors."""
    
    def __init__(
        self,
        message: str,
        expected_seq_num: Optional[int] = None,
        actual_seq_num: Optional[int] = None,
        **kwargs
    ):
        """Initialize sequence exception.
        
        Args:
            message: Error message
            expected_seq_num: Expected sequence number
            actual_seq_num: Actual sequence number
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if expected_seq_num:
            context['expected_seq_num'] = expected_seq_num
        if actual_seq_num:
            context['actual_seq_num'] = actual_seq_num
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.expected_seq_num = expected_seq_num
        self.actual_seq_num = actual_seq_num


class ChecksumException(FixProtocolException):
    """Exception for checksum validation errors."""
    
    def __init__(
        self,
        message: str,
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None,
        **kwargs
    ):
        """Initialize checksum exception.
        
        Args:
            message: Error message
            expected_checksum: Expected checksum
            actual_checksum: Actual checksum
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        if expected_checksum:
            context['expected_checksum'] = expected_checksum
        if actual_checksum:
            context['actual_checksum'] = actual_checksum
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum


# Exception handling utilities
class ExceptionHandler:
    """Utility class for handling exceptions in a structured way."""
    
    def __init__(self):
        """Initialize exception handler."""
        self.handlers = {}
        self.default_handler = None
    
    def register_handler(self, exception_type: type, handler: callable) -> None:
        """Register a handler for a specific exception type.
        
        Args:
            exception_type: Exception type to handle
            handler: Handler function that takes the exception as argument
        """
        self.handlers[exception_type] = handler
    
    def set_default_handler(self, handler: callable) -> None:
        """Set default handler for unregistered exception types.
        
        Args:
            handler: Default handler function
        """
        self.default_handler = handler
    
    def handle_exception(self, exception: Exception) -> Any:
        """Handle an exception using registered handlers.
        
        Args:
            exception: Exception to handle
            
        Returns:
            Result from handler function
        """
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self.handlers:
            return self.handlers[exception_type](exception)
        
        # Look for parent class matches
        for registered_type, handler in self.handlers.items():
            if isinstance(exception, registered_type):
                return handler(exception)
        
        # Use default handler if available
        if self.default_handler:
            return self.default_handler(exception)
        
        # Re-raise if no handler found
        raise exception


def create_exception_from_dict(exception_data: Dict[str, Any]) -> FixTesterException:
    """Create exception from dictionary representation.
    
    Args:
        exception_data: Dictionary containing exception details
        
    Returns:
        FixTesterException instance
    """
    exception_type = exception_data.get('type', 'FixTesterException')
    message = exception_data.get('message', 'Unknown error')
    error_code = exception_data.get('error_code')
    context = exception_data.get('context', {})
    
    # Map exception type names to classes
    exception_classes = {
        'FixTesterException': FixTesterException,
        'FixEngineException': FixEngineException,
        'ConnectionException': ConnectionException,
        'SessionException': SessionException,
        'MessageException': MessageException,
        'ValidationException': ValidationException,
        'ConfigurationException': ConfigurationException,
        'TestException': TestException,
        'TimeoutException': TimeoutException,
        'AuthenticationException': AuthenticationException,
        'AuthorizationException': AuthorizationException,
        'DataException': DataException,
        'ResourceException': ResourceException,
        'FixProtocolException': FixProtocolException,
        'FixVersionException': FixVersionException,
        'SequenceException': SequenceException,
        'ChecksumException': ChecksumException,
    }
    
    exception_class = exception_classes.get(exception_type, FixTesterException)
    
    return exception_class(
        message=message,
        error_code=error_code,
        context=context
    )


# Error codes for categorization
class ErrorCodes:
    """Standard error codes for FixTester exceptions."""
    
    # Connection errors
    CONNECTION_FAILED = "CONN_001"
    CONNECTION_TIMEOUT = "CONN_002"
    CONNECTION_REFUSED = "CONN_003"
    CONNECTION_LOST = "CONN_004"
    
    # Session errors
    SESSION_NOT_FOUND = "SESS_001"
    SESSION_ALREADY_EXISTS = "SESS_002"
    SESSION_LOGON_FAILED = "SESS_003"
    SESSION_LOGOUT_FAILED = "SESS_004"
    SESSION_STATE_INVALID = "SESS_005"
    
    # Message errors
    MESSAGE_INVALID = "MSG_001"
    MESSAGE_TIMEOUT = "MSG_002"
    MESSAGE_DUPLICATE = "MSG_003"
    MESSAGE_OUT_OF_ORDER = "MSG_004"
    MESSAGE_MISSING_FIELD = "MSG_005"
    MESSAGE_INVALID_FIELD = "MSG_006"
    
    # Validation errors
    VALIDATION_FAILED = "VAL_001"
    VALIDATION_MISSING_REQUIRED = "VAL_002"
    VALIDATION_INVALID_FORMAT = "VAL_003"
    VALIDATION_OUT_OF_RANGE = "VAL_004"
    VALIDATION_BUSINESS_RULE = "VAL_005"
    
    # Configuration errors
    CONFIG_NOT_FOUND = "CFG_001"
    CONFIG_INVALID_FORMAT = "CFG_002"
    CONFIG_MISSING_REQUIRED = "CFG_003"
    CONFIG_INVALID_VALUE = "CFG_004"
    
    # Test errors
    TEST_FAILED = "TEST_001"
    TEST_TIMEOUT = "TEST_002"
    TEST_SETUP_FAILED = "TEST_003"
    TEST_TEARDOWN_FAILED = "TEST_004"
    TEST_DATA_INVALID = "TEST_005"
    
    # Authentication/Authorization errors
    AUTH_FAILED = "AUTH_001"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_002"
    AUTH_TOKEN_EXPIRED = "AUTH_003"
    AUTH_TOKEN_INVALID = "AUTH_004"
    
    # Data errors
    DATA_NOT_FOUND = "DATA_001"
    DATA_INVALID_FORMAT = "DATA_002"
    DATA_CORRUPTION = "DATA_003"
    DATA_ACCESS_DENIED = "DATA_004"
    
    # Resource errors
    RESOURCE_NOT_FOUND = "RES_001"
    RESOURCE_UNAVAILABLE = "RES_002"
    RESOURCE_EXHAUSTED = "RES_003"
    RESOURCE_LOCKED = "RES_004"
    
    # FIX Protocol specific errors
    FIX_VERSION_MISMATCH = "FIX_001"
    FIX_SEQUENCE_ERROR = "FIX_002"
    FIX_CHECKSUM_ERROR = "FIX_003"
    FIX_DUPLICATE_MESSAGE = "FIX_004"
    FIX_UNSUPPORTED_MESSAGE = "FIX_005"


# Context managers for exception handling
class exception_context:
    """Context manager for adding context to exceptions."""
    
    def __init__(self, **context):
        """Initialize exception context.
        
        Args:
            **context: Context information to add to exceptions
        """
        self.context = context
        
    def __enter__(self):
        """Enter the exception context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Add context to any exception that occurred."""
        if exc_type and isinstance(exc_val, FixTesterException):
            # Add context to FixTester exceptions
            exc_val.context.update(self.context)
        elif exc_type and exc_val:
            # Wrap other exceptions in FixTesterException with context
            wrapped_exception = FixTesterException(
                message=str(exc_val),
                context=self.context,
                cause=exc_val
            )
            # Replace the exception
            raise wrapped_exception from exc_val
        
        return False  # Don't suppress the exception


class error_boundary:
    """Context manager for error boundary handling."""
    
    def __init__(
        self,
        error_handler: Optional[callable] = None,
        reraise: bool = True,
        log_errors: bool = True
    ):
        """Initialize error boundary.
        
        Args:
            error_handler: Optional function to handle errors
            reraise: Whether to reraise exceptions after handling
            log_errors: Whether to log caught exceptions
        """
        self.error_handler = error_handler
        self.reraise = reraise
        self.log_errors = log_errors
        self.exception = None
        
    def __enter__(self):
        """Enter the error boundary."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle any exception that occurred."""
        if exc_type and exc_val:
            self.exception = exc_val
            
            # Log the error if requested
            if self.log_errors:
                from fixtester.utils.logger import setup_logger
                logger = setup_logger(__name__)
                logger.error(f"Error in boundary: {exc_val}", exc_info=True)
            
            # Call error handler if provided
            if self.error_handler:
                try:
                    self.error_handler(exc_val)
                except Exception as handler_error:
                    # Don't let handler errors break the boundary
                    if self.log_errors:
                        logger.error(f"Error in error handler: {handler_error}")
            
            # Suppress exception if not reraising
            return not self.reraise
        
        return False


# Utility functions for exception handling
def safe_execute(func: callable, *args, default=None, **kwargs):
    """Safely execute a function, returning default on exception.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        default: Default value to return on exception
        **kwargs: Keyword arguments for function
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


def collect_exceptions(funcs: list, continue_on_error: bool = True) -> tuple:
    """Execute multiple functions and collect any exceptions.
    
    Args:
        funcs: List of functions to execute
        continue_on_error: Whether to continue after exceptions
        
    Returns:
        Tuple of (results, exceptions)
    """
    results = []
    exceptions = []
    
    for func in funcs:
        try:
            if callable(func):
                result = func()
            else:
                result = func[0](*func[1:])  # Assume (func, *args) format
            results.append(result)
        except Exception as e:
            exceptions.append(e)
            results.append(None)
            
            if not continue_on_error:
                break
    
    return results, exceptions


def retry_on_exception(
    func: callable,
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """Retry function on specific exceptions.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        exceptions: Exceptions to retry on
        delay: Initial delay between attempts
        backoff_factor: Backoff multiplier
        
    Returns:
        Function result
        
    Raises:
        Last exception if all attempts fail
    """
    import time
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                time.sleep(current_delay)
                current_delay *= backoff_factor
    
    raise last_exception


def exception_to_http_status(exception: Exception) -> int:
    """Convert exception to appropriate HTTP status code.
    
    Args:
        exception: Exception to convert
        
    Returns:
        HTTP status code
    """
    if isinstance(exception, AuthenticationException):
        return 401
    elif isinstance(exception, AuthorizationException):
        return 403
    elif isinstance(exception, (ResourceException, DataException)):
        if "not found" in str(exception).lower():
            return 404
        else:
            return 400
    elif isinstance(exception, ValidationException):
        return 400
    elif isinstance(exception, TimeoutException):
        return 408
    elif isinstance(exception, ConfigurationException):
        return 500
    else:
        return 500  # Internal server error for unknown exceptions


# Global exception registry for tracking common errors
class ExceptionRegistry:
    """Registry for tracking and analyzing exceptions."""
    
    def __init__(self):
        """Initialize exception registry."""
        self.exceptions = []
        self.exception_counts = {}
        self.lock = threading.Lock()
    
    def register_exception(self, exception: Exception) -> None:
        """Register an exception occurrence.
        
        Args:
            exception: Exception to register
        """
        import time
        
        with self.lock:
            exception_record = {
                'type': type(exception).__name__,
                'message': str(exception),
                'timestamp': time.time(),
                'context': getattr(exception, 'context', {}),
                'error_code': getattr(exception, 'error_code', None)
            }
            
            self.exceptions.append(exception_record)
            
            # Update counts
            exception_type = type(exception).__name__
            self.exception_counts[exception_type] = self.exception_counts.get(exception_type, 0) + 1
            
            # Keep only recent exceptions (last 1000)
            if len(self.exceptions) > 1000:
                self.exceptions = self.exceptions[-1000:]
    
    def get_exception_stats(self) -> Dict[str, Any]:
        """Get exception statistics.
        
        Returns:
            Dictionary containing exception statistics
        """
        with self.lock:
            return {
                'total_exceptions': len(self.exceptions),
                'exception_counts': self.exception_counts.copy(),
                'recent_exceptions': self.exceptions[-10:]  # Last 10 exceptions
            }
    
    def clear_registry(self) -> None:
        """Clear the exception registry."""
        with self.lock:
            self.exceptions.clear()
            self.exception_counts.clear()


# Global exception registry instance
_global_registry = ExceptionRegistry()


def get_exception_registry() -> ExceptionRegistry:
    """Get the global exception registry.
    
    Returns:
        Global ExceptionRegistry instance
    """
    return _global_registry


# Custom exception formatter
def format_exception_chain(exception: Exception, include_cause: bool = True) -> str:
    """Format exception with its cause chain.
    
    Args:
        exception: Exception to format
        include_cause: Whether to include cause exceptions
        
    Returns:
        Formatted exception string
    """
    lines = []
    current = exception
    
    while current:
        lines.append(f"{type(current).__name__}: {current}")
        
        if hasattr(current, 'context') and current.context:
            context_str = ", ".join(f"{k}={v}" for k, v in current.context.items())
            lines.append(f"  Context: {context_str}")
        
        if include_cause and hasattr(current, 'cause') and current.cause:
            current = current.cause
            lines.append("  Caused by:")
        else:
            break
    
    return "\n".join(lines)