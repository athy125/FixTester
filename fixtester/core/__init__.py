"""
Core module for FixTester containing the main engine components.

This module provides:
- FixEngine: Main FIX protocol engine for managing connections and messages
- MessageFactory: Factory for creating FIX messages with different types and templates
- SessionManager: Manager for handling multiple FIX sessions and their lifecycle
- MessageValidator: Comprehensive validator for FIX protocol messages
"""

from .engine import FixEngine, FixApplication
from .message_factory import MessageFactory
from .session_manager import SessionManager, SessionInfo, SessionState
from .validator import (
    MessageValidator, 
    ValidationReport, 
    ValidationIssue, 
    ValidationResult, 
    ValidationLevel
)

__all__ = [
    "FixEngine",
    "FixApplication", 
    "MessageFactory",
    "SessionManager",
    "SessionInfo",
    "SessionState",
    "MessageValidator",
    "ValidationReport",
    "ValidationIssue", 
    "ValidationResult",
    "ValidationLevel"
]

# Version information
__version__ = "0.1.0"
__author__ = "FixTester Team"
__description__ = "Core components for automated FIX protocol testing"