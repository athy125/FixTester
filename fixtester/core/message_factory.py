"""
Message factory for creating FIX protocol messages with different versions and templates.
"""
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Union

import quickfix as fix

from fixtester.utils.logger import setup_logger


class MessageFactory:
    """Factory for creating FIX messages of different types and versions."""

    def __init__(self, protocol_version: str = "FIX.4.4", templates_dir: Optional[str] = None):
        """Initialize the message factory.

        Args:
            protocol_version: FIX protocol version (e.g., "FIX.4.2", "FIX.4.4", "FIX.5.0")
            templates_dir: Directory containing message templates (default: config/templates)
        """
        self.protocol_version = protocol_version
        self.templates_dir = templates_dir or os.path.join("config", "templates")
        self.logger = setup_logger(__name__)
        
        # Load message templates if available
        self.templates = self._load_templates()
        
        # Map FIX versions to their respective classes
        self.fix_version_map = {
            "FIX.4.2": fix.BeginString_FIX42,
            "FIX.4.4": fix.BeginString_FIX44,
            "FIX.5.0": fix.BeginString_FIXT11
        }
        
        # Map message types to their respective classes
        self.message_type_map = {
            # Admin messages
            "Heartbeat": {"type": fix.MsgType_Heartbeat, "class": fix.Heartbeat},
            "TestRequest": {"type": fix.MsgType_TestRequest, "class": fix.TestRequest},
            "Logon": {"type": fix.MsgType_Logon, "class": fix.Logon},
            "Logout": {"type": fix.MsgType_Logout, "class": fix.Logout},
            "Reject": {"type": fix.MsgType_Reject, "class": fix.Reject},
            
            # Application messages
            "NewOrderSingle": {"type": fix.MsgType_NewOrderSingle, "class": fix.NewOrderSingle},
            "OrderCancelRequest": {"type": fix.MsgType_OrderCancelRequest, "class": fix.OrderCancelRequest},
            "OrderCancelReplaceRequest": {"type": fix.MsgType_OrderCancelReplaceRequest, "class": fix.OrderCancelReplaceRequest},
            "OrderStatusRequest": {"type": fix.MsgType_OrderStatusRequest, "class": fix.OrderStatusRequest},
            "ExecutionReport": {"type": fix.MsgType_ExecutionReport, "class": fix.ExecutionReport},
            "OrderCancelReject": {"type": fix.MsgType_OrderCancelReject, "class": fix.OrderCancelReject},
            "MarketDataRequest": {"type": fix.MsgType_MarketDataRequest, "class": fix.MarketDataRequest},
            "MarketDataSnapshotFullRefresh": {"type": fix.MsgType_MarketDataSnapshotFullRefresh, "class": fix.MarketDataSnapshotFullRefresh},
            "MarketDataIncrementalRefresh": {"type": fix.MsgType_MarketDataIncrementalRefresh, "class": fix.MarketDataIncrementalRefresh},
        }

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load message templates from files.

        Returns:
            Dictionary of templates by message type
        """
        templates = {}
        
        # Determine template file name based on protocol version
        version_num = self.protocol_version.replace(".", "").lower()
        template_file = os.path.join(self.templates_dir, f"{version_num}_templates.json")
        
        # Try to load JSON templates if they exist
        try:
            if os.path.exists(template_file):
                with open(template_file, 'r') as f:
                    templates = json.load(f)
                self.logger.info(f"Loaded templates from {template_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load templates from {template_file}: {e}")
            
        return templates

    def create_message(self, message_type: str, fields: Dict[str, Any] = None) -> fix.Message:
        """Create a FIX message of the specified type.

        Args:
            message_type: FIX message type name (e.g., "NewOrderSingle")
            fields: Dictionary of field values to set in the message

        Returns:
            FIX message object

        Raises:
            ValueError: If the message type is unknown
        """
        if message_type not in self.message_type_map:
            raise ValueError(f"Unknown message type: {message_type}")
            
        # Get the message type info
        msg_info = self.message_type_map[message_type]
        
        # Create a new message
        message = msg_info["class"]()
        header = message.getHeader()
        
        # Set basic header fields
        header.setField(fix.BeginString(self.fix_version_map.get(self.protocol_version, fix.BeginString_FIX44)))
        header.setField(fix.MsgType(msg_info["type"]))
        
        # Apply template if available
        if message_type in self.templates:
            self._apply_template(message, self.templates[message_type])
            
        # Set fields provided by the caller
        if fields:
            self._set_fields(message, fields)
            
        return message

    def create_from_template(self, template_name: str, fields: Dict[str, Any] = None) -> fix.Message:
        """Create a message from a named template.

        Args:
            template_name: Name of the template
            fields: Additional fields to override the template

        Returns:
            FIX message object

        Raises:
            ValueError: If the template is not found
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
            
        template = self.templates[template_name]
        message_type = template.get("type")
        
        if not message_type:
            raise ValueError(f"Template missing message type: {template_name}")
            
        # Create base message from the template's type
        message = self.create_message(message_type, template.get("fields", {}))
        
        # Override with provided fields
        if fields:
            self._set_fields(message, fields)
            
        return message

    def _apply_template(self, message: fix.Message, template: Dict[str, Any]) -> None:
        """Apply a template to a message.

        Args:
            message: FIX message to modify
            template: Template data
        """
        if "fields" in template and isinstance(template["fields"], dict):
            self._set_fields(message, template["fields"])

    def _set_fields(self, message: fix.Message, fields: Dict[str, Any]) -> None:
        """Set multiple fields in a message.

        Args:
            message: FIX message to modify
            fields: Dictionary of field values
        """
        for field_name, value in fields.items():
            self._set_field(message, field_name, value)

    def _set_field(self, message: fix.Message, field_name: str, value: Any) -> None:
        """Set a single field in a message.

        Args:
            message: FIX message to modify
            field_name: Field name or number
            value: Field value
        """
        # Handle common FIX fields
        if field_name == "ClOrdID":
            message.setField(fix.ClOrdID(str(value)))
        elif field_name == "OrderID":
            message.setField(fix.OrderID(str(value)))
        elif field_name == "Symbol":
            message.setField(fix.Symbol(str(value)))
        elif field_name == "Side":
            message.setField(fix.Side(str(value)))
        elif field_name == "OrdType":
            message.setField(fix.OrdType(str(value)))
        elif field_name == "Price":
            message.setField(fix.Price(float(value)))
        elif field_name == "OrderQty":
            message.setField(fix.OrderQty(float(value)))
        elif field_name == "TimeInForce":
            message.setField(fix.TimeInForce(str(value)))
        elif field_name == "TransactTime":
            # Handle datetime - accept datetime object or string
            if isinstance(value, datetime.datetime):
                dt = value
            elif isinstance(value, str):
                try:
                    dt = datetime.datetime.strptime(value, "%Y%m%d-%H:%M:%S.%f")
                except ValueError:
                    try:
                        dt = datetime.datetime.strptime(value, "%Y%m%d-%H:%M:%S")
                    except ValueError:
                        dt = datetime.datetime.now()
            else:
                dt = datetime.datetime.now()
                
            message.setField(fix.TransactTime(dt.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]))
        else:
            # Try to set as a generic field
            try:
                # Check if field_name is a number
                if isinstance(field_name, int) or field_name.isdigit():
                    # Set by tag number
                    field_num = int(field_name)
                    message.setField(field_num, str(value))
                else:
                    # Look up the field by name using fix module attributes
                    field_attr = getattr(fix, field_name, None)
                    if field_attr is not None:
                        message.setField(field_attr(str(value)))
                    else:
                        self.logger.warning(f"Unknown field: {field_name}")
            except Exception as e:
                self.logger.error(f"Error setting field {field_name}: {e}")

    def get_field(self, message: fix.Message, field_name: str) -> Optional[str]:
        """Get a field value from a message.

        Args:
            message: FIX message
            field_name: Field name or number

        Returns:
            Field value as string, or None if not found
        """
        try:
            # Check if field_name is a number
            if isinstance(field_name, int) or field_name.isdigit():
                # Get by tag number
                field_num = int(field_name)
                return message.getField(field_num)
            else:
                # Look up the field by name using fix module attributes
                field_attr = getattr(fix, field_name, None)
                if field_attr is not None:
                    field = field_attr()
                    message.getField(field)
                    return field.getValue()
                else:
                    self.logger.warning(f"Unknown field: {field_name}")
                    return None
        except Exception as e:
            self.logger.debug(f"Field {field_name} not found: {e}")
            return None

    def message_to_dict(self, message: fix.Message) -> Dict[str, str]:
        """Convert a FIX message to a dictionary.

        Args:
            message: FIX message

        Returns:
            Dictionary of field tag to value
        """
        result = {}
        
        # Get string representation
        msg_str = message.toString()
        
        # Parse the message string (format: 8=FIX.4.4␁9=100␁35=D␁...)
        fields = msg_str.split("\x01")
        
        for field in fields:
            if not field:
                continue
                
            # Split at the first = to get tag and value
            parts = field.split('=', 1)
            if len(parts) == 2:
                tag, value = parts
                result[tag] = value
                
        return result

    def generate_unique_id(self, prefix: str = "ID") -> str:
        """Generate a unique ID for a message.

        Args:
            prefix: Prefix for the ID

        Returns:
            Unique ID string
        """
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        random_part = os.urandom(4).hex()
        return f"{prefix}-{timestamp}-{random_part}"