"""
Helper utilities and common functions for FixTester including data conversion,
string manipulation, file operations, and other utility functions.
"""
import os
import re
import json
import hashlib
import secrets
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple
from pathlib import Path
import uuid

from fixtester.utils.logger import setup_logger


class DataConverter:
    """Utility class for data type conversions and formatting."""

    @staticmethod
    def to_fix_timestamp(dt: Optional[datetime] = None) -> str:
        """Convert datetime to FIX timestamp format (YYYYMMDD-HH:MM:SS).

        Args:
            dt: Datetime object (defaults to current UTC time)

        Returns:
            FIX formatted timestamp string
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y%m%d-%H:%M:%S")

    @staticmethod
    def from_fix_timestamp(timestamp_str: str) -> datetime:
        """Parse FIX timestamp format to datetime object.

        Args:
            timestamp_str: FIX formatted timestamp string

        Returns:
            Datetime object in UTC

        Raises:
            ValueError: If timestamp format is invalid
        """
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d-%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(f"Invalid FIX timestamp format: {timestamp_str}") from e

    @staticmethod
    def to_fix_date(dt: Optional[datetime] = None) -> str:
        """Convert datetime to FIX date format (YYYYMMDD).

        Args:
            dt: Datetime object (defaults to current UTC time)

        Returns:
            FIX formatted date string
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y%m%d")

    @staticmethod
    def from_fix_date(date_str: str) -> datetime:
        """Parse FIX date format to datetime object.

        Args:
            date_str: FIX formatted date string

        Returns:
            Datetime object in UTC

        Raises:
            ValueError: If date format is invalid
        """
        try:
            return datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(f"Invalid FIX date format: {date_str}") from e

    @staticmethod
    def to_fix_time(dt: Optional[datetime] = None) -> str:
        """Convert datetime to FIX time format (HH:MM:SS).

        Args:
            dt: Datetime object (defaults to current UTC time)

        Returns:
            FIX formatted time string
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%H:%M:%S")

    @staticmethod
    def from_fix_time(time_str: str) -> datetime:
        """Parse FIX time format to datetime object (today's date).

        Args:
            time_str: FIX formatted time string

        Returns:
            Datetime object in UTC with today's date

        Raises:
            ValueError: If time format is invalid
        """
        try:
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            today = datetime.now(timezone.utc).date()
            return datetime.combine(today, time_obj, timezone.utc)
        except ValueError as e:
            raise ValueError(f"Invalid FIX time format: {time_str}") from e

    @staticmethod
    def format_decimal(value: Union[int, float], precision: int = 2) -> str:
        """Format decimal value for FIX messages.

        Args:
            value: Numeric value to format
            precision: Number of decimal places

        Returns:
            Formatted decimal string
        """
        return f"{float(value):.{precision}f}"

    @staticmethod
    def parse_fix_message_to_dict(message_str: str) -> Dict[str, str]:
        """Parse FIX message string to dictionary.

        Args:
            message_str: FIX message string with SOH delimiters

        Returns:
            Dictionary mapping field numbers to values
        """
        # Replace SOH (ASCII 1) with pipe for easier parsing
        normalized = message_str.replace('\x01', '|')
        
        result = {}
        for field in normalized.split('|'):
            if '=' in field:
                tag, value = field.split('=', 1)
                result[tag] = value
        
        return result

    @staticmethod
    def dict_to_fix_message(fields: Dict[str, str]) -> str:
        """Convert dictionary to FIX message string.

        Args:
            fields: Dictionary mapping field numbers to values

        Returns:
            FIX message string with SOH delimiters
        """
        parts = []
        for tag, value in fields.items():
            parts.append(f"{tag}={value}")
        
        return '\x01'.join(parts) + '\x01'


class StringUtils:
    """String manipulation utilities."""

    @staticmethod
    def generate_order_id(prefix: str = "", length: int = 10) -> str:
        """Generate a unique order ID.

        Args:
            prefix: Optional prefix for the order ID
            length: Length of the random part

        Returns:
            Generated order ID
        """
        random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(length))
        return f"{prefix}{random_part}" if prefix else random_part

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize a string for use as a filename.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(' .')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed"
        
        return sanitized

    @staticmethod
    def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate string to maximum length with optional suffix.

        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add when truncating

        Returns:
            Truncated string
        """
        if len(text) <= max_length:
            return text
        
        truncate_length = max_length - len(suffix)
        if truncate_length <= 0:
            return suffix[:max_length]
        
        return text[:truncate_length] + suffix

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert CamelCase to snake_case.

        Args:
            name: CamelCase string

        Returns:
            snake_case string
        """
        # Insert underscore before uppercase letters that follow lowercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before uppercase letters that follow lowercase letters or digits
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @staticmethod
    def snake_to_camel(name: str, capitalize_first: bool = False) -> str:
        """Convert snake_case to CamelCase.

        Args:
            name: snake_case string
            capitalize_first: Whether to capitalize the first letter

        Returns:
            CamelCase string
        """
        components = name.split('_')
        if capitalize_first:
            return ''.join(word.capitalize() for word in components)
        else:
            return components[0] + ''.join(word.capitalize() for word in components[1:])

    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """Mask sensitive data showing only first/last characters.

        Args:
            data: Sensitive data to mask
            visible_chars: Number of characters to show at start and end

        Returns:
            Masked string
        """
        if len(data) <= visible_chars * 2:
            return '*' * len(data)
        
        start = data[:visible_chars]
        end = data[-visible_chars:]
        middle = '*' * (len(data) - visible_chars * 2)
        
        return f"{start}{middle}{end}"


class FileUtils:
    """File and directory utilities."""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Path object for the directory
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def safe_file_write(filepath: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
        """Safely write content to file with atomic operation.

        Args:
            filepath: File path
            content: Content to write
            encoding: File encoding
        """
        filepath = Path(filepath)
        temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            # Write to temporary file first
            with open(temp_filepath, 'w', encoding=encoding) as f:
                f.write(content)
            
            # Atomically move to final location
            temp_filepath.replace(filepath)
            
        except Exception:
            # Clean up temporary file on error
            if temp_filepath.exists():
                temp_filepath.unlink()
            raise

    @staticmethod
    def get_file_hash(filepath: Union[str, Path], algorithm: str = 'sha256') -> str:
        """Calculate hash of file contents.

        Args:
            filepath: File path
            algorithm: Hash algorithm (md5, sha1, sha256, etc.)

        Returns:
            Hex string of file hash

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If algorithm is not supported
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            hasher = hashlib.new(algorithm)
        except ValueError:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()

    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = True) -> List[Path]:
        """Find files matching pattern in directory.

        Args:
            directory: Directory to search in
            pattern: Glob pattern to match
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        directory = Path(directory)
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))

    @staticmethod
    def clean_old_files(directory: Union[str, Path], max_age_days: int, pattern: str = "*") -> int:
        """Clean old files from directory.

        Args:
            directory: Directory to clean
            max_age_days: Maximum age of files to keep
            pattern: File pattern to match

        Returns:
            Number of files deleted
        """
        import time
        
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        deleted_count = 0
        
        for filepath in directory.glob(pattern):
            if filepath.is_file() and filepath.stat().st_mtime < cutoff_time:
                try:
                    filepath.unlink()
                    deleted_count += 1
                except OSError:
                    pass  # File might be in use
        
        return deleted_count

    @staticmethod
    def backup_file(filepath: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create a backup copy of a file.

        Args:
            filepath: File to backup
            backup_dir: Directory for backup (defaults to same directory)

        Returns:
            Path to backup file

        Raises:
            FileNotFoundError: If source file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")
        
        if backup_dir:
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{filepath.suffix}"
        else:
            backup_path = filepath.with_name(f"{filepath.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{filepath.suffix}")
        
        import shutil
        shutil.copy2(filepath, backup_path)
        
        return backup_path


class ValidationUtils:
    """Validation utilities for data integrity checks."""

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Check if email address is valid.

        Args:
            email: Email address to validate

        Returns:
            True if valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        return bool(re.match(pattern, email))

    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """Check if trading symbol is valid.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic symbol validation - alphanumeric and dots only
        pattern = r'^[A-Z0-9.]{1,12}
        return bool(re.match(pattern, symbol.upper()))

    @staticmethod
    def is_valid_price(price: Union[str, float, int]) -> bool:
        """Check if price value is valid.

        Args:
            price: Price value to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            price_float = float(price)
            return price_float > 0 and price_float < 1000000
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_valid_quantity(quantity: Union[str, int, float]) -> bool:
        """Check if quantity value is valid.

        Args:
            quantity: Quantity value to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            qty_float = float(quantity)
            return qty_float > 0 and qty_float == int(qty_float)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_config_dict(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate configuration dictionary has required keys.

        Args:
            config: Configuration dictionary
            required_keys: List of required keys

        Returns:
            List of missing keys
        """
        if not isinstance(config, dict):
            return required_keys
        
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        
        return missing_keys


class RetryUtils:
    """Utilities for retry logic and error handling."""

    @staticmethod
    def retry_with_backoff(
        func: callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple = (Exception,)
    ) -> Any:
        """Retry function with exponential backoff.

        Args:
            func: Function to retry
            max_attempts: Maximum number of attempts
            base_delay: Initial delay between attempts
            max_delay: Maximum delay between attempts
            backoff_factor: Backoff multiplier
            exceptions: Tuple of exceptions to catch

        Returns:
            Function result

        Raises:
            Exception: Last exception if all attempts fail
        """
        import time
        
        last_exception = None
        delay = base_delay
        
        for attempt in range(max_attempts):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_attempts - 1:
                    break
                
                # Wait before retry
                time.sleep(min(delay, max_delay))
                delay *= backoff_factor
        
        raise last_exception

    @staticmethod
    def timeout_wrapper(func: callable, timeout_seconds: float) -> Any:
        """Execute function with timeout.

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds

        Returns:
            Function result

        Raises:
            TimeoutError: If function doesn't complete within timeout
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
        
        # Set up signal handler (Unix only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func()
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
                    result[0] = func()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]


class CryptoUtils:
    """Cryptographic utilities for security operations."""

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token.

        Args:
            length: Length of the token

        Returns:
            Secure random token string
        """
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt.

        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        import hashlib
        
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash.

        Args:
            password: Password to verify
            hashed_password: Stored hash
            salt: Salt used for hashing

        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = CryptoUtils.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, hashed_password)


class NetworkUtils:
    """Network-related utilities."""

    @staticmethod
    def is_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
        """Check if a port is open on a host.

        Args:
            host: Hostname or IP address
            port: Port number
            timeout: Connection timeout

        Returns:
            True if port is open, False otherwise
        """
        import socket
        
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, socket.error):
            return False

    @staticmethod
    def get_local_ip() -> str:
        """Get local IP address.

        Returns:
            Local IP address string
        """
        import socket
        
        try:
            # Connect to a remote server to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    @staticmethod
    def validate_host_port(host: str, port: Union[str, int]) -> bool:
        """Validate host and port combination.

        Args:
            host: Hostname or IP address
            port: Port number

        Returns:
            True if valid, False otherwise
        """
        try:
            port_int = int(port)
            if not (1 <= port_int <= 65535):
                return False
            
            # Basic hostname validation
            if not host or len(host) > 253:
                return False
            
            return True
        except (ValueError, TypeError):
            return False


class BatchProcessor:
    """Utility for processing items in batches."""

    def __init__(self, batch_size: int = 100):
        """Initialize batch processor.

        Args:
            batch_size: Size of each batch
        """
        self.batch_size = batch_size
        self.logger = setup_logger(__name__)

    def process_in_batches(
        self,
        items: List[Any],
        process_func: callable,
        progress_callback: Optional[callable] = None
    ) -> List[Any]:
        """Process items in batches.

        Args:
            items: List of items to process
            process_func: Function to process each batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of results from all batches
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                
                if progress_callback:
                    progress_callback(batch_num, total_batches, len(batch))
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num}: {e}")
                raise
        
        return results


class ConfigMerger:
    """Utility for merging configuration dictionaries."""

    @staticmethod
    def deep_merge(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Configuration to merge in

        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = ConfigMerger.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    @staticmethod
    def flatten_config(config: Dict[str, Any], prefix: str = "", separator: str = ".") -> Dict[str, Any]:
        """Flatten nested configuration dictionary.

        Args:
            config: Configuration dictionary to flatten
            prefix: Prefix for keys
            separator: Key separator

        Returns:
            Flattened configuration dictionary
        """
        result = {}
        
        for key, value in config.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(ConfigMerger.flatten_config(value, new_key, separator))
            else:
                result[new_key] = value
        
        return result


# Utility functions for common operations
def generate_uuid() -> str:
    """Generate a UUID string.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def current_timestamp() -> float:
    """Get current timestamp.

    Returns:
        Current timestamp as float
    """
    return datetime.now(timezone.utc).timestamp()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def parse_size_string(size_str: str) -> int:
    """Parse size string with units (e.g., '10MB', '1GB').

    Args:
        size_str: Size string with optional unit

    Returns:
        Size in bytes

    Raises:
        ValueError: If format is invalid
    """
    size_str = size_str.strip().upper()
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4
    }
    
    # Extract number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([A-Z]*B?), size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    
    number, unit = match.groups()
    unit = unit or 'B'
    
    if unit not in units:
        raise ValueError(f"Unknown size unit: {unit}")
    
    return int(float(number) * units[unit])


def chunk_list(lst: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """Split list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Yields:
        Chunks of the original list
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely cast value to target type with default fallback.

    Args:
        value: Value to cast
        target_type: Target type
        default: Default value if casting fails

    Returns:
        Cast value or default
    """
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default