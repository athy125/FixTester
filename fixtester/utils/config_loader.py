"""
Configuration loader utility for loading and validating YAML configuration files.
"""
import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

from fixtester.utils.logger import setup_logger


class ConfigLoader:
    """Utility class for loading and validating configuration files."""

    def __init__(self):
        """Initialize the config loader."""
        self.logger = setup_logger(__name__)

    @classmethod
    def load(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML is invalid
        """
        loader = cls()
        return loader._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file with validation.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # Apply environment variable substitution
            config = self._substitute_env_vars(config)
            
            # Validate basic structure
            self._validate_config(config)
            
            self.logger.debug(f"Configuration loaded successfully: {len(config)} top-level keys")
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in configuration file {config_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            raise

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration values.

        Args:
            config: Configuration value (dict, list, or primitive)

        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string_env_vars(config)
        else:
            return config

    def _substitute_string_env_vars(self, value: str) -> str:
        """Substitute environment variables in a string value.

        Supports formats:
        - ${VAR_NAME} - Required variable
        - ${VAR_NAME:-default} - Variable with default value

        Args:
            value: String value potentially containing environment variables

        Returns:
            String with environment variables substituted
        """
        import re
        
        # Pattern for ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else None
            
            # Get environment variable value
            env_value = os.environ.get(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # Required variable not found
                raise ValueError(f"Required environment variable '{var_name}' not found")
        
        return re.sub(pattern, replace_var, value)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the basic structure of the configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If the configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate FIX configuration if present
        if 'fix' in config:
            self._validate_fix_config(config['fix'])
        
        # Validate logging configuration if present
        if 'logging' in config:
            self._validate_logging_config(config['logging'])
        
        # Validate validation configuration if present
        if 'validation' in config:
            self._validate_validation_config(config['validation'])

    def _validate_fix_config(self, fix_config: Dict[str, Any]) -> None:
        """Validate FIX protocol configuration.

        Args:
            fix_config: FIX configuration dictionary

        Raises:
            ValueError: If the FIX configuration is invalid
        """
        if not isinstance(fix_config, dict):
            raise ValueError("FIX configuration must be a dictionary")
        
        # Validate connection settings
        if 'connection' in fix_config:
            conn_config = fix_config['connection']
            if not isinstance(conn_config, dict):
                raise ValueError("FIX connection configuration must be a dictionary")
            
            required_fields = ['sender_comp_id', 'target_comp_id']
            for field in required_fields:
                if field not in conn_config:
                    self.logger.warning(f"Missing recommended FIX connection field: {field}")
            
            # Validate port if present
            if 'port' in conn_config:
                port = conn_config['port']
                if not isinstance(port, int) or port <= 0 or port > 65535:
                    raise ValueError(f"Invalid port number: {port}")
        
        # Validate FIX version if present
        if 'version' in fix_config:
            version = fix_config['version']
            valid_versions = ['FIX.4.0', 'FIX.4.1', 'FIX.4.2', 'FIX.4.3', 'FIX.4.4', 'FIX.5.0', 'FIXT.1.1']
            if version not in valid_versions:
                self.logger.warning(f"Unrecognized FIX version: {version}")

    def _validate_logging_config(self, logging_config: Dict[str, Any]) -> None:
        """Validate logging configuration.

        Args:
            logging_config: Logging configuration dictionary

        Raises:
            ValueError: If the logging configuration is invalid
        """
        if not isinstance(logging_config, dict):
            raise ValueError("Logging configuration must be a dictionary")
        
        # Validate log level if present
        if 'level' in logging_config:
            level = logging_config['level'].upper()
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                raise ValueError(f"Invalid log level: {level}")

    def _validate_validation_config(self, validation_config: Dict[str, Any]) -> None:
        """Validate message validation configuration.

        Args:
            validation_config: Validation configuration dictionary

        Raises:
            ValueError: If the validation configuration is invalid
        """
        if not isinstance(validation_config, dict):
            raise ValueError("Validation configuration must be a dictionary")
        
        # Validate validation level if present
        if 'level' in validation_config:
            level = validation_config['level'].lower()
            valid_levels = ['strict', 'normal', 'permissive']
            if level not in valid_levels:
                raise ValueError(f"Invalid validation level: {level}")

    @classmethod
    def merge_configs(cls, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.

        Later configs override earlier ones for conflicting keys.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            if not isinstance(config, dict):
                continue
            merged = cls._deep_merge(merged, config)
        
        return merged

    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    @classmethod
    def save_config(cls, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the configuration
        """
        loader = cls()
        loader._save_config(config, output_path)

    def _save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to file.

        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the configuration
        """
        output_file = Path(output_path)
        
        # Create directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to {output_path}: {e}")
            raise

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get a default configuration template.

        Returns:
            Default configuration dictionary
        """
        return {
            "fix": {
                "version": "FIX.4.4",
                "connection": {
                    "host": "localhost",
                    "port": 9878,
                    "sender_comp_id": "CLIENT1",
                    "target_comp_id": "SERVER1",
                    "heartbeat_interval": 30
                }
            },
            "logging": {
                "level": "INFO",
                "file": "logs/fixtester.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "validation": {
                "level": "normal",
                "strict_mode": False,
                "required_fields_check": True,
                "value_range_check": True,
                "format_check": True,
                "business_logic_check": True
            },
            "session_pool": {
                "max_sessions": 10,
                "timeout": 300,
                "heartbeat_interval": 30
            }
        }