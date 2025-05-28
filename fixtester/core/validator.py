"""
Message validator for FIX protocol messages with comprehensive validation rules,
field checking, and customizable validation policies.
"""
import re
import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import quickfix as fix

from fixtester.utils.logger import setup_logger


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a message."""
    level: ValidationResult
    field_name: Optional[str]
    field_value: Optional[str]
    message: str
    rule_name: str


@dataclass
class ValidationReport:
    """Complete validation report for a message."""
    message_type: str
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int
    validated_at: datetime.datetime


class MessageValidator:
    """Validates FIX protocol messages according to configurable rules."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the message validator.

        Args:
            config: Validation configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logger(__name__)
        
        # Validation settings
        self.validation_level = ValidationLevel(
            self.config.get("level", "normal")
        )
        self.strict_mode = self.config.get("strict_mode", False)
        self.required_fields_check = self.config.get("required_fields_check", True)
        self.value_range_check = self.config.get("value_range_check", True)
        self.format_check = self.config.get("format_check", True)
        self.business_logic_check = self.config.get("business_logic_check", True)
        
        # Custom validation rules
        self.custom_rules = self.config.get("custom_rules", {})
        
        # Initialize validation rules
        self._init_validation_rules()

    def _init_validation_rules(self) -> None:
        """Initialize built-in validation rules."""
        # Required fields by message type
        self.required_fields = {
            "NewOrderSingle": ["ClOrdID", "Symbol", "Side", "TransactTime", "OrdType"],
            "OrderCancelRequest": ["OrigClOrdID", "ClOrdID", "Symbol", "Side", "TransactTime"],
            "OrderCancelReplaceRequest": ["OrigClOrdID", "ClOrdID", "Symbol", "Side", "TransactTime", "OrdType"],
            "ExecutionReport": ["OrderID", "ExecID", "ExecType", "OrdStatus", "Symbol", "Side", "LeavesQty", "CumQty"],
            "MarketDataRequest": ["MDReqID", "SubscriptionRequestType", "MarketDepth"],
            "OrderStatusRequest": ["ClOrdID", "Symbol", "Side"],
        }
        
        # Field value constraints
        self.field_constraints = {
            "Side": {"type": "enum", "values": ["1", "2"]},  # 1=Buy, 2=Sell
            "OrdType": {"type": "enum", "values": ["1", "2", "3", "4", "5", "6"]},  # Market, Limit, Stop, etc.
            "TimeInForce": {"type": "enum", "values": ["0", "1", "2", "3", "4", "6"]},  # Day, GTC, etc.
            "ExecType": {"type": "enum", "values": ["0", "1", "2", "4", "5", "6", "8", "C", "F"]},
            "OrdStatus": {"type": "enum", "values": ["0", "1", "2", "4", "5", "6", "8", "9", "A", "C", "E"]},
            "SubscriptionRequestType": {"type": "enum", "values": ["0", "1", "2"]},  # Snapshot, Subscribe, Unsubscribe
            "MDEntryType": {"type": "enum", "values": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]},
            
            # Numeric constraints
            "OrderQty": {"type": "positive_number"},
            "Price": {"type": "positive_number"},
            "StopPx": {"type": "positive_number"},
            "LastQty": {"type": "non_negative_number"},
            "LastPx": {"type": "positive_number"},
            "LeavesQty": {"type": "non_negative_number"},
            "CumQty": {"type": "non_negative_number"},
            "AvgPx": {"type": "non_negative_number"},
            "MarketDepth": {"type": "non_negative_integer"},
            
            # String format constraints
            "ClOrdID": {"type": "string", "min_length": 1, "max_length": 64},
            "OrderID": {"type": "string", "min_length": 1, "max_length": 64},
            "ExecID": {"type": "string", "min_length": 1, "max_length": 64},
            "Symbol": {"type": "string", "min_length": 1, "max_length": 32, "pattern": r"^[A-Z0-9\.]+$"},
            "Currency": {"type": "string", "length": 3, "pattern": r"^[A-Z]{3}$"},
            "Account": {"type": "string", "max_length": 32},
            "Text": {"type": "string", "max_length": 255},
            
            # Date/time constraints
            "TransactTime": {"type": "utc_timestamp"},
            "SendingTime": {"type": "utc_timestamp"},
            "MDEntryDate": {"type": "date"},
            "MDEntryTime": {"type": "time"},
        }
        
        # Business logic rules
        self.business_rules = {
            "limit_order_requires_price": {
                "condition": lambda msg: self._get_field_value(msg, "OrdType") == "2",
                "requirement": lambda msg: self._has_field(msg, "Price"),
                "message": "Limit orders must specify a price"
            },
            "stop_order_requires_stop_price": {
                "condition": lambda msg: self._get_field_value(msg, "OrdType") in ["3", "4"],
                "requirement": lambda msg: self._has_field(msg, "StopPx"),
                "message": "Stop orders must specify a stop price"
            },
            "execution_report_quantities": {
                "condition": lambda msg: self._get_message_type(msg) == "ExecutionReport",
                "requirement": lambda msg: self._validate_execution_quantities(msg),
                "message": "ExecutionReport quantities are inconsistent (CumQty + LeavesQty should equal OrderQty)"
            },
            "market_data_subscription_fields": {
                "condition": lambda msg: (self._get_message_type(msg) == "MarketDataRequest" and 
                                        self._get_field_value(msg, "SubscriptionRequestType") == "1"),
                "requirement": lambda msg: self._has_field(msg, "NoMDEntryTypes") and self._has_field(msg, "NoRelatedSym"),
                "message": "Market data subscription requests must specify MD entry types and symbols"
            }
        }

    def validate_message(self, message: fix.Message, expected_type: Optional[str] = None) -> ValidationReport:
        """Validate a FIX message comprehensively.

        Args:
            message: FIX message to validate
            expected_type: Expected message type (optional)

        Returns:
            ValidationReport containing all validation results
        """
        issues = []
        
        # Get message type
        try:
            msg_type = self._get_message_type(message)
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                field_name="MsgType",
                field_value=None,
                message=f"Could not determine message type: {e}",
                rule_name="message_type_check"
            ))
            return self._create_report("Unknown", issues)
        
        # Check expected message type
        if expected_type and msg_type != expected_type:
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                field_name="MsgType",
                field_value=msg_type,
                message=f"Expected message type '{expected_type}', got '{msg_type}'",
                rule_name="expected_type_check"
            ))
        
        # Validate required fields
        if self.required_fields_check:
            issues.extend(self._validate_required_fields(message, msg_type))
        
        # Validate field values
        if self.value_range_check:
            issues.extend(self._validate_field_values(message))
        
        # Validate field formats
        if self.format_check:
            issues.extend(self._validate_field_formats(message))
        
        # Validate business logic
        if self.business_logic_check:
            issues.extend(self._validate_business_logic(message))
        
        # Apply custom rules
        issues.extend(self._apply_custom_rules(message, msg_type))
        
        return self._create_report(msg_type, issues)

    def _validate_required_fields(self, message: fix.Message, msg_type: str) -> List[ValidationIssue]:
        """Validate that all required fields are present.

        Args:
            message: FIX message to validate
            msg_type: Message type

        Returns:
            List of validation issues
        """
        issues = []
        required_fields = self.required_fields.get(msg_type, [])
        
        for field_name in required_fields:
            if not self._has_field(message, field_name):
                level = ValidationResult.ERROR if self.strict_mode else ValidationResult.WARNING
                issues.append(ValidationIssue(
                    level=level,
                    field_name=field_name,
                    field_value=None,
                    message=f"Required field '{field_name}' is missing",
                    rule_name="required_fields"
                ))
        
        return issues

    def _validate_field_values(self, message: fix.Message) -> List[ValidationIssue]:
        """Validate field values against constraints.

        Args:
            message: FIX message to validate

        Returns:
            List of validation issues
        """
        issues = []
        
        # Get all fields in the message
        field_map = {}
        iterator = message.iterator()
        while iterator.hasNext():
            field = iterator.next()
            field_num = field.getTag()
            field_value = field.getString()
            field_name = self._get_field_name(field_num)
            if field_name:
                field_map[field_name] = field_value
        
        # Validate each field
        for field_name, field_value in field_map.items():
            if field_name in self.field_constraints:
                constraint = self.field_constraints[field_name]
                field_issues = self._validate_field_constraint(field_name, field_value, constraint)
                issues.extend(field_issues)
        
        return issues

    def _validate_field_constraint(self, field_name: str, field_value: str, constraint: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate a single field against its constraint.

        Args:
            field_name: Name of the field
            field_value: Value of the field
            constraint: Constraint definition

        Returns:
            List of validation issues
        """
        issues = []
        constraint_type = constraint.get("type")
        
        try:
            if constraint_type == "enum":
                if field_value not in constraint["values"]:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        field_name=field_name,
                        field_value=field_value,
                        message=f"Invalid value '{field_value}' for field '{field_name}'. Valid values: {constraint['values']}",
                        rule_name="enum_constraint"
                    ))
            
            elif constraint_type == "positive_number":
                try:
                    num_val = float(field_value)
                    if num_val <= 0:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=field_name,
                            field_value=field_value,
                            message=f"Field '{field_name}' must be a positive number, got '{field_value}'",
                            rule_name="positive_number_constraint"
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        field_name=field_name,
                        field_value=field_value,
                        message=f"Field '{field_name}' must be a valid number, got '{field_value}'",
                        rule_name="numeric_constraint"
                    ))
            
            elif constraint_type == "non_negative_number":
                try:
                    num_val = float(field_value)
                    if num_val < 0:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=field_name,
                            field_value=field_value,
                            message=f"Field '{field_name}' must be non-negative, got '{field_value}'",
                            rule_name="non_negative_constraint"
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        field_name=field_name,
                        field_value=field_value,
                        message=f"Field '{field_name}' must be a valid number, got '{field_value}'",
                        rule_name="numeric_constraint"
                    ))
            
            elif constraint_type == "non_negative_integer":
                try:
                    int_val = int(field_value)
                    if int_val < 0:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=field_name,
                            field_value=field_value,
                            message=f"Field '{field_name}' must be a non-negative integer, got '{field_value}'",
                            rule_name="non_negative_integer_constraint"
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        field_name=field_name,
                        field_value=field_value,
                        message=f"Field '{field_name}' must be a valid integer, got '{field_value}'",
                        rule_name="integer_constraint"
                    ))
        
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                field_name=field_name,
                field_value=field_value,
                message=f"Error validating field constraint: {e}",
                rule_name="constraint_validation_error"
            ))
        
        return issues

    def _validate_field_formats(self, message: fix.Message) -> List[ValidationIssue]:
        """Validate field formats (length, pattern, etc.).

        Args:
            message: FIX message to validate

        Returns:
            List of validation issues
        """
        issues = []
        
        # Get all fields in the message
        field_map = {}
        iterator = message.iterator()
        while iterator.hasNext():
            field = iterator.next()
            field_num = field.getTag()
            field_value = field.getString()
            field_name = self._get_field_name(field_num)
            if field_name:
                field_map[field_name] = field_value
        
        # Validate formats
        for field_name, field_value in field_map.items():
            if field_name in self.field_constraints:
                constraint = self.field_constraints[field_name]
                
                # Check string length constraints
                if constraint.get("type") == "string":
                    if "min_length" in constraint and len(field_value) < constraint["min_length"]:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=field_name,
                            field_value=field_value,
                            message=f"Field '{field_name}' is too short (minimum {constraint['min_length']} characters)",
                            rule_name="min_length_constraint"
                        ))
                    
                    if "max_length" in constraint and len(field_value) > constraint["max_length"]:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=field_name,
                            field_value=field_value,
                            message=f"Field '{field_name}' is too long (maximum {constraint['max_length']} characters)",
                            rule_name="max_length_constraint"
                        ))
                    
                    if "length" in constraint and len(field_value) != constraint["length"]:
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=field_name,
                            field_value=field_value,
                            message=f"Field '{field_name}' must be exactly {constraint['length']} characters",
                            rule_name="exact_length_constraint"
                        ))
                    
                    if "pattern" in constraint:
                        if not re.match(constraint["pattern"], field_value):
                            issues.append(ValidationIssue(
                                level=ValidationResult.ERROR,
                                field_name=field_name,
                                field_value=field_value,
                                message=f"Field '{field_name}' does not match required pattern",
                                rule_name="pattern_constraint"
                            ))
        
        return issues

    def _validate_business_logic(self, message: fix.Message) -> List[ValidationIssue]:
        """Validate business logic rules.

        Args:
            message: FIX message to validate

        Returns:
            List of validation issues
        """
        issues = []
        
        for rule_name, rule in self.business_rules.items():
            try:
                # Check if rule condition applies
                if rule["condition"](message):
                    # Check if requirement is met
                    if not rule["requirement"](message):
                        issues.append(ValidationIssue(
                            level=ValidationResult.ERROR,
                            field_name=None,
                            field_value=None,
                            message=rule["message"],
                            rule_name=rule_name
                        ))
            except Exception as e:
                self.logger.warning(f"Error validating business rule '{rule_name}': {e}")
        
        return issues

    def _apply_custom_rules(self, message: fix.Message, msg_type: str) -> List[ValidationIssue]:
        """Apply custom validation rules.

        Args:
            message: FIX message to validate
            msg_type: Message type

        Returns:
            List of validation issues
        """
        issues = []
        
        # Apply message-type specific custom rules
        if msg_type in self.custom_rules:
            rules = self.custom_rules[msg_type]
            for rule_name, rule_func in rules.items():
                try:
                    result = rule_func(message)
                    if isinstance(result, list):
                        issues.extend(result)
                    elif isinstance(result, ValidationIssue):
                        issues.append(result)
                except Exception as e:
                    self.logger.warning(f"Error applying custom rule '{rule_name}': {e}")
        
        return issues

    def _create_report(self, msg_type: str, issues: List[ValidationIssue]) -> ValidationReport:
        """Create a validation report from issues.

        Args:
            msg_type: Message type
            issues: List of validation issues

        Returns:
            ValidationReport
        """
        errors_count = sum(1 for issue in issues if issue.level == ValidationResult.ERROR)
        warnings_count = sum(1 for issue in issues if issue.level == ValidationResult.WARNING)
        is_valid = errors_count == 0
        
        return ValidationReport(
            message_type=msg_type,
            is_valid=is_valid,
            issues=issues,
            warnings_count=warnings_count,
            errors_count=errors_count,
            validated_at=datetime.datetime.utcnow()
        )

    def _get_message_type(self, message: fix.Message) -> str:
        """Get the message type from a FIX message.

        Args:
            message: FIX message

        Returns:
            Message type string
        """
        msg_type = message.getHeader().getField(fix.MsgType())
        
        # Map message type codes to names
        type_map = {
            "0": "Heartbeat",
            "1": "TestRequest",
            "2": "ResendRequest",
            "3": "Reject",
            "4": "SequenceReset",
            "5": "Logout",
            "A": "Logon",
            "D": "NewOrderSingle",
            "F": "OrderCancelRequest",
            "G": "OrderCancelReplaceRequest",
            "H": "OrderStatusRequest",
            "8": "ExecutionReport",
            "9": "OrderCancelReject",
            "V": "MarketDataRequest",
            "W": "MarketDataSnapshotFullRefresh",
            "X": "MarketDataIncrementalRefresh",
        }
        
        return type_map.get(msg_type, f"Unknown({msg_type})")

    def _has_field(self, message: fix.Message, field_name: str) -> bool:
        """Check if a message has a specific field.

        Args:
            message: FIX message
            field_name: Field name

        Returns:
            True if field exists, False otherwise
        """
        try:
            field_num = self._get_field_number(field_name)
            return message.isSetField(field_num)
        except:
            return False

    def _get_field_value(self, message: fix.Message, field_name: str) -> Optional[str]:
        """Get field value from a message.

        Args:
            message: FIX message
            field_name: Field name

        Returns:
            Field value or None if not found
        """
        try:
            field_num = self._get_field_number(field_name)
            return message.getField(field_num)
        except:
            return None

    def _get_field_number(self, field_name: str) -> int:
        """Get FIX field number from field name.

        Args:
            field_name: Field name

        Returns:
            Field number
        """
        # Field name to number mapping (same as in message_factory.py)
        field_map = {
            "BeginString": fix.FIELD_BeginString,
            "MsgType": fix.FIELD_MsgType,
            "ClOrdID": fix.FIELD_ClOrdID,
            "OrderID": fix.FIELD_OrderID,
            "ExecID": fix.FIELD_ExecID,
            "ExecType": fix.FIELD_ExecType,
            "OrdStatus": fix.FIELD_OrdStatus,
            "Symbol": fix.FIELD_Symbol,
            "Side": fix.FIELD_Side,
            "OrderQty": fix.FIELD_OrderQty,
            "OrdType": fix.FIELD_OrdType,
            "Price": fix.FIELD_Price,
            "StopPx": fix.FIELD_StopPx,
            "TimeInForce": fix.FIELD_TimeInForce,
            "TransactTime": fix.FIELD_TransactTime,
            "OrigClOrdID": fix.FIELD_OrigClOrdID,
            "CumQty": fix.FIELD_CumQty,
            "LeavesQty": fix.FIELD_LeavesQty,
            "LastQty": fix.FIELD_LastQty,
            "LastPx": fix.FIELD_LastPx,
            "AvgPx": fix.FIELD_AvgPx,
            "MDReqID": fix.FIELD_MDReqID,
            "SubscriptionRequestType": fix.FIELD_SubscriptionRequestType,
            "MarketDepth": fix.FIELD_MarketDepth,
            "NoMDEntryTypes": fix.FIELD_NoMDEntryTypes,
            "MDEntryType": fix.FIELD_MDEntryType,
            "NoRelatedSym": fix.FIELD_NoRelatedSym,
            "Account": fix.FIELD_Account,
            "Currency": fix.FIELD_Currency,
            "Text": fix.FIELD_Text,
            "SendingTime": fix.FIELD_SendingTime,
            "MDEntryDate": fix.FIELD_MDEntryDate,
            "MDEntryTime": fix.FIELD_MDEntryTime,
        }
        
        if field_name in field_map:
            return field_map[field_name]
        else:
            try:
                return int(field_name)
            except ValueError:
                raise ValueError(f"Unknown field name: {field_name}")

    def _get_field_name(self, field_num: int) -> Optional[str]:
        """Get field name from FIX field number.

        Args:
            field_num: Field number

        Returns:
            Field name or None if not found
        """
        # Reverse mapping
        field_map = {
            fix.FIELD_BeginString: "BeginString",
            fix.FIELD_MsgType: "MsgType",
            fix.FIELD_ClOrdID: "ClOrdID",
            fix.FIELD_OrderID: "OrderID",
            fix.FIELD_ExecID: "ExecID",
            fix.FIELD_ExecType: "ExecType",
            fix.FIELD_OrdStatus: "OrdStatus",
            fix.FIELD_Symbol: "Symbol",
            fix.FIELD_Side: "Side",
            fix.FIELD_OrderQty: "OrderQty",
            fix.FIELD_OrdType: "OrdType",
            fix.FIELD_Price: "Price",
            fix.FIELD_StopPx: "StopPx",
            fix.FIELD_TimeInForce: "TimeInForce",
            fix.FIELD_TransactTime: "TransactTime",
            fix.FIELD_OrigClOrdID: "OrigClOrdID",
            fix.FIELD_CumQty: "CumQty",
            fix.FIELD_LeavesQty: "LeavesQty",
            fix.FIELD_LastQty: "LastQty",
            fix.FIELD_LastPx: "LastPx",
            fix.FIELD_AvgPx: "AvgPx",
            fix.FIELD_MDReqID: "MDReqID",
            fix.FIELD_SubscriptionRequestType: "SubscriptionRequestType",
            fix.FIELD_MarketDepth: "MarketDepth",
            fix.FIELD_NoMDEntryTypes: "NoMDEntryTypes",
            fix.FIELD_MDEntryType: "MDEntryType",
            fix.FIELD_NoRelatedSym: "NoRelatedSym",
            fix.FIELD_Account: "Account",
            fix.FIELD_Currency: "Currency",
            fix.FIELD_Text: "Text",
            fix.FIELD_SendingTime: "SendingTime",
            fix.FIELD_MDEntryDate: "MDEntryDate",
            fix.FIELD_MDEntryTime: "MDEntryTime",
        }
        
        return field_map.get(field_num)

    def _validate_execution_quantities(self, message: fix.Message) -> bool:
        """Validate execution report quantities consistency.

        Args:
            message: ExecutionReport message

        Returns:
            True if quantities are consistent, False otherwise
        """
        try:
            order_qty = float(self._get_field_value(message, "OrderQty") or "0")
            cum_qty = float(self._get_field_value(message, "CumQty") or "0")
            leaves_qty = float(self._get_field_value(message, "LeavesQty") or "0")
            
            # For execution reports, CumQty + LeavesQty should equal OrderQty
            tolerance = 0.000001  # Small tolerance for floating point comparison
            return abs((cum_qty + leaves_qty) - order_qty) <= tolerance
        except:
            return True  # If we can't validate, assume it's okay

    def add_custom_rule(self, message_type: str, rule_name: str, rule_func: callable) -> None:
        """Add a custom validation rule.

        Args:
            message_type: Message type to apply the rule to
            rule_name: Name of the rule
            rule_func: Function that takes a message and returns ValidationIssue(s) or None
        """
        if message_type not in self.custom_rules:
            self.custom_rules[message_type] = {}
        
        self.custom_rules[message_type][rule_name] = rule_func
        self.logger.info(f"Added custom rule '{rule_name}' for message type '{message_type}'")

    def remove_custom_rule(self, message_type: str, rule_name: str) -> bool:
        """Remove a custom validation rule.

        Args:
            message_type: Message type
            rule_name: Name of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if (message_type in self.custom_rules and 
            rule_name in self.custom_rules[message_type]):
            del self.custom_rules[message_type][rule_name]
            self.logger.info(f"Removed custom rule '{rule_name}' for message type '{message_type}'")
            return True
        return False

    def get_validation_summary(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """Get summary statistics from multiple validation reports.

        Args:
            reports: List of validation reports

        Returns:
            Summary statistics dictionary
        """
        total_messages = len(reports)
        valid_messages = sum(1 for r in reports if r.is_valid)
        total_errors = sum(r.errors_count for r in reports)
        total_warnings = sum(r.warnings_count for r in reports)
        
        message_type_stats = {}
        for report in reports:
            msg_type = report.message_type
            if msg_type not in message_type_stats:
                message_type_stats[msg_type] = {"count": 0, "valid": 0, "errors": 0, "warnings": 0}
            
            message_type_stats[msg_type]["count"] += 1
            if report.is_valid:
                message_type_stats[msg_type]["valid"] += 1
            message_type_stats[msg_type]["errors"] += report.errors_count
            message_type_stats[msg_type]["warnings"] += report.warnings_count
        
        return {
            "total_messages": total_messages,
            "valid_messages": valid_messages,
            "invalid_messages": total_messages - valid_messages,
            "validation_rate": valid_messages / total_messages if total_messages > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "message_type_stats": message_type_stats,
        }