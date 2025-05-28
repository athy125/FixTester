"""
Main FixTester engine module responsible for establishing and managing FIX sessions,
sending and receiving messages, and coordinating the testing workflow.
"""
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Union

import quickfix as fix

from fixtester.core.validator import MessageValidator
from fixtester.utils.config_loader import ConfigLoader
from fixtester.utils.logger import setup_logger


class FixApplication(fix.Application):
    """QuickFIX application implementation for handling FIX messages."""

    def __init__(self, engine: 'FixEngine'):
        """Initialize the FIX application.

        Args:
            engine: The parent FixEngine instance
        """
        super().__init__()
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self.message_queue = queue.Queue()
        self.sessions = {}
        self.response_events = {}

    def onCreate(self, session_id: fix.SessionID) -> None:
        """Called when a new session is created.

        Args:
            session_id: The FIX session ID
        """
        self.logger.info(f"Session created: {session_id}")
        self.sessions[str(session_id)] = session_id

    def onLogon(self, session_id: fix.SessionID) -> None:
        """Called when a session successfully logs on.

        Args:
            session_id: The FIX session ID
        """
        self.logger.info(f"Logon successful: {session_id}")
        self.engine.on_session_event("logon", str(session_id))

    def onLogout(self, session_id: fix.SessionID) -> None:
        """Called when a session logs out.

        Args:
            session_id: The FIX session ID
        """
        self.logger.info(f"Logout: {session_id}")
        self.engine.on_session_event("logout", str(session_id))

    def toAdmin(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Called before sending an administrative message.

        Args:
            message: The FIX message
            session_id: The FIX session ID
        """
        msg_type = message.getHeader().getField(fix.MsgType())
        self.logger.debug(f"Admin message sent ({msg_type}): {message}")

    def fromAdmin(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Called when an administrative message is received.

        Args:
            message: The FIX message
            session_id: The FIX session ID
        """
        msg_type = message.getHeader().getField(fix.MsgType())
        self.logger.debug(f"Admin message received ({msg_type}): {message}")

    def toApp(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Called before sending an application message.

        Args:
            message: The FIX message
            session_id: The FIX session ID
        """
        self.logger.info(f"App message sent: {message}")

    def fromApp(self, message: fix.Message, session_id: fix.SessionID) -> None:
        """Called when an application message is received.

        Args:
            message: The FIX message
            session_id: The FIX session ID
        """
        self.logger.info(f"App message received: {message}")
        
        # Put the message in the queue for processing
        self.message_queue.put((message, str(session_id)))
        
        # Check if there's a waiting response event for this message
        msg_type = message.getHeader().getField(fix.MsgType())
        clord_id = None
        
        # Try to extract ClOrdID if it exists
        try:
            clord_id = message.getField(fix.ClOrdID())
        except fix.FieldNotFound:
            pass
            
        # Notify any waiting threads
        if clord_id and clord_id in self.response_events:
            self.response_events[clord_id].set()


class FixEngine:
    """Main FIX engine for managing sessions and message processing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the FIX engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__, config.get("logging", {}))
        self.application = FixApplication(self)
        self.settings = self._create_settings()
        self.store_factory = fix.FileStoreFactory(self.settings)
        self.log_factory = fix.FileLogFactory(self.settings)
        self.initiator = None
        self.validator = MessageValidator(config.get("validation", {}))
        self.active_sessions = {}
        self.is_running = False
        self.message_handlers = {}

    @classmethod
    def from_config(cls, config_path: str) -> 'FixEngine':
        """Create a FixEngine instance from a configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            A new FixEngine instance
        """
        config = ConfigLoader.load(config_path)
        return cls(config)

    def _create_settings(self) -> fix.SessionSettings:
        """Create QuickFIX session settings from the configuration.

        Returns:
            QuickFIX SessionSettings
        """
        fix_config = self.config.get("fix", {})
        connection = fix_config.get("connection", {})
        
        settings = fix.SessionSettings()
        
        # Default settings
        settings.set(fix.Dictionary())
        settings.set("ConnectionType", "initiator")
        settings.set("ReconnectInterval", "30")
        settings.set("FileLogPath", "logs")
        settings.set("FileStorePath", "store")
        settings.set("StartTime", "00:00:00")
        settings.set("EndTime", "23:59:59")
        settings.set("UseDataDictionary", "Y")
        
        # Session specific settings
        session = fix.Dictionary()
        session.setString("BeginString", fix_config.get("version", "FIX.4.4"))
        session.setString("SenderCompID", connection.get("sender_comp_id", "CLIENT"))
        session.setString("TargetCompID", connection.get("target_comp_id", "SERVER"))
        session.setString("SocketConnectHost", connection.get("host", "localhost"))
        session.setString("SocketConnectPort", str(connection.get("port", 9878)))
        session.setString("HeartBtInt", str(connection.get("heartbeat_interval", 30)))
        
        # Path to data dictionary for the specific FIX version
        fix_version = fix_config.get("version", "FIX.4.4")
        data_dict_path = f"spec/{fix_version.replace('.', '')}.xml"
        session.setString("DataDictionary", data_dict_path)
        
        # Add the session to settings
        settings.set(fix.SessionID(
            fix_config.get("version", "FIX.4.4"),
            connection.get("sender_comp_id", "CLIENT"),
            connection.get("target_comp_id", "SERVER")
        ), session)
        
        return settings

    def start(self) -> None:
        """Start the FIX engine and initiate connections."""
        if self.is_running:
            self.logger.warning("Engine is already running")
            return
            
        self.logger.info("Starting FIX engine")
        
        # Create and start the initiator
        self.initiator = fix.SocketInitiator(
            self.application,
            self.store_factory,
            self.settings,
            self.log_factory
        )
        
        self.initiator.start()
        self.is_running = True
        
        # Start message processing thread
        self.processing_thread = threading.Thread(
            target=self._process_messages,
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("FIX engine started")

    def stop(self) -> None:
        """Stop the FIX engine and close all connections."""
        if not self.is_running:
            self.logger.warning("Engine is not running")
            return
            
        self.logger.info("Stopping FIX engine")
        
        if self.initiator:
            self.initiator.stop()
            
        self.is_running = False
        self.logger.info("FIX engine stopped")

    def on_session_event(self, event_type: str, session_id: str) -> None:
        """Handle session events (logon, logout).

        Args:
            event_type: Type of the session event
            session_id: The session ID as a string
        """
        if event_type == "logon":
            self.active_sessions[session_id] = {
                "connected_at": time.time(),
                "status": "active"
            }
        elif event_type == "logout":
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "inactive"

    def send_message(self, message: fix.Message, session_id: Optional[str] = None) -> bool:
        """Send a FIX message.

        Args:
            message: The FIX message to send
            session_id: Optional session ID (uses the first active session if None)

        Returns:
            True if the message was sent successfully, False otherwise
        """
        if not self.is_running:
            self.logger.error("Cannot send message: Engine is not running")
            return False
            
        if not session_id:
            # Use first active session if none specified
            active = [sid for sid, info in self.active_sessions.items() 
                    if info["status"] == "active"]
            if not active:
                self.logger.error("No active sessions available")
                return False
            session_id = active[0]
            
        # Get the SessionID object from the string representation
        if session_id not in self.application.sessions:
            self.logger.error(f"Session not found: {session_id}")
            return False
            
        session = self.application.sessions[session_id]
        
        # Send the message
        return fix.Session.sendToTarget(message, session)

    def send_and_wait(
        self, 
        message: fix.Message, 
        timeout: float = 10.0,
        session_id: Optional[str] = None
    ) -> Optional[fix.Message]:
        """Send a message and wait for a response.

        Args:
            message: The FIX message to send
            timeout: Maximum time to wait for a response in seconds
            session_id: Optional session ID

        Returns:
            The response message or None if timed out
        """
        # Extract ClOrdID from the message
        try:
            clord_id = message.getField(fix.ClOrdID())
        except fix.FieldNotFound:
            self.logger.error("Message must contain ClOrdID field")
            return None
            
        # Create an event for this message
        response_event = threading.Event()
        self.application.response_events[clord_id] = response_event
        
        # Send the message
        if not self.send_message(message, session_id):
            self.logger.error("Failed to send message")
            del self.application.response_events[clord_id]
            return None
            
        # Wait for the response
        if not response_event.wait(timeout):
            self.logger.warning(f"Timed out waiting for response to {clord_id}")
            del self.application.response_events[clord_id]
            return None
            
        # Find the response message in the queue
        try:
            # Wait for the message to be processed
            time.sleep(0.1)  # Give a small time for processing
            
            # Look through recent messages
            while not self.application.message_queue.empty():
                msg, _ = self.application.message_queue.get(block=False)
                try:
                    if msg.getField(fix.ClOrdID()) == clord_id:
                        return msg
                except fix.FieldNotFound:
                    pass
                    
            self.logger.warning(f"Response event triggered but message not found for {clord_id}")
            return None
        finally:
            # Clean up
            del self.application.response_events[clord_id]

    def register_message_handler(self, msg_type: str, handler: callable) -> None:
        """Register a handler for a specific message type.

        Args:
            msg_type: FIX message type (e.g., "D" for NewOrderSingle)
            handler: Callback function to handle the message
        """
        self.message_handlers[msg_type] = handler
        self.logger.debug(f"Registered handler for message type {msg_type}")

    def _process_messages(self) -> None:
        """Process incoming messages in a separate thread."""
        while self.is_running:
            try:
                # Get a message from the queue (blocking with timeout)
                try:
                    message, session_id = self.application.message_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                # Get message type
                msg_type = message.getHeader().getField(fix.MsgType())
                
                # Call the appropriate handler if registered
                if msg_type in self.message_handlers:
                    try:
                        self.message_handlers[msg_type](message, session_id)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
                        
                # Mark the task as done
                self.application.message_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in message processing: {e}")