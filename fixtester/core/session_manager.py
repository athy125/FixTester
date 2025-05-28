"""
Session manager for handling multiple FIX sessions, connection pooling,
and session lifecycle management.
"""
import threading
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

import quickfix as fix

from fixtester.utils.logger import setup_logger


class SessionState(Enum):
    """Enumeration of possible session states."""
    CREATED = "created"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOGGED_ON = "logged_on"
    LOGGING_OUT = "logging_out"
    LOGGED_OUT = "logged_out"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class SessionInfo:
    """Information about a FIX session."""
    session_id: str
    fix_version: str
    sender_comp_id: str
    target_comp_id: str
    host: str
    port: int
    state: SessionState
    created_at: float
    connected_at: Optional[float] = None
    last_heartbeat: Optional[float] = None
    message_count_sent: int = 0
    message_count_received: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class SessionManager:
    """Manages multiple FIX sessions and their lifecycle."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the session manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(__name__, config.get("logging", {}))
        self.sessions: Dict[str, SessionInfo] = {}
        self.session_lock = threading.RLock()
        self.active_sessions: Set[str] = set()
        self.session_callbacks: Dict[str, List[callable]] = {}
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Session pool configuration
        self.max_sessions = config.get("session_pool", {}).get("max_sessions", 10)
        self.session_timeout = config.get("session_pool", {}).get("timeout", 300)
        self.heartbeat_interval = config.get("session_pool", {}).get("heartbeat_interval", 30)

    def create_session(
        self,
        session_id: str,
        fix_version: str,
        sender_comp_id: str,
        target_comp_id: str,
        host: str,
        port: int,
        **kwargs
    ) -> bool:
        """Create a new FIX session.

        Args:
            session_id: Unique session identifier
            fix_version: FIX protocol version
            sender_comp_id: Sender CompID
            target_comp_id: Target CompID
            host: Connection host
            port: Connection port
            **kwargs: Additional session parameters

        Returns:
            True if session was created successfully, False otherwise
        """
        with self.session_lock:
            if session_id in self.sessions:
                self.logger.warning(f"Session {session_id} already exists")
                return False
                
            if len(self.sessions) >= self.max_sessions:
                self.logger.error(f"Maximum number of sessions ({self.max_sessions}) reached")
                return False
                
            # Create session info
            session_info = SessionInfo(
                session_id=session_id,
                fix_version=fix_version,
                sender_comp_id=sender_comp_id,
                target_comp_id=target_comp_id,
                host=host,
                port=port,
                state=SessionState.CREATED,
                created_at=time.time()
            )
            
            self.sessions[session_id] = session_info
            self.session_callbacks[session_id] = []
            
            self.logger.info(f"Created session {session_id}")
            return True

    def connect_session(self, session_id: str) -> bool:
        """Connect a session.

        Args:
            session_id: Session identifier

        Returns:
            True if connection was initiated successfully, False otherwise
        """
        with self.session_lock:
            if session_id not in self.sessions:
                self.logger.error(f"Session {session_id} not found")
                return False
                
            session_info = self.sessions[session_id]
            
            if session_info.state not in [SessionState.CREATED, SessionState.DISCONNECTED]:
                self.logger.warning(f"Session {session_id} is in state {session_info.state}, cannot connect")
                return False
                
            # Update session state
            session_info.state = SessionState.CONNECTING
            
            self.logger.info(f"Connecting session {session_id}")
            self._notify_session_callbacks(session_id, "connecting", session_info)
            
            return True

    def disconnect_session(self, session_id: str, graceful: bool = True) -> bool:
        """Disconnect a session.

        Args:
            session_id: Session identifier
            graceful: Whether to perform graceful logout

        Returns:
            True if disconnection was initiated successfully, False otherwise
        """
        with self.session_lock:
            if session_id not in self.sessions:
                self.logger.error(f"Session {session_id} not found")
                return False
                
            session_info = self.sessions[session_id]
            
            if graceful and session_info.state == SessionState.LOGGED_ON:
                session_info.state = SessionState.LOGGING_OUT
                self.logger.info(f"Gracefully logging out session {session_id}")
            else:
                session_info.state = SessionState.DISCONNECTED
                self.logger.info(f"Disconnecting session {session_id}")
                
            # Remove from active sessions
            self.active_sessions.discard(session_id)
            
            self._notify_session_callbacks(session_id, "disconnecting", session_info)
            
            return True

    def remove_session(self, session_id: str) -> bool:
        """Remove a session completely.

        Args:
            session_id: Session identifier

        Returns:
            True if session was removed successfully, False otherwise
        """
        with self.session_lock:
            if session_id not in self.sessions:
                self.logger.warning(f"Session {session_id} not found")
                return False
                
            # Disconnect first if needed
            session_info = self.sessions[session_id]
            if session_info.state in [SessionState.CONNECTED, SessionState.LOGGED_ON]:
                self.disconnect_session(session_id, graceful=False)
                
            # Remove from all tracking structures
            del self.sessions[session_id]
            del self.session_callbacks[session_id]
            self.active_sessions.discard(session_id)
            
            self.logger.info(f"Removed session {session_id}")
            return True

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get information about a session.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo object or None if session not found
        """
        with self.session_lock:
            return self.sessions.get(session_id)

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs.

        Returns:
            List of active session identifiers
        """
        with self.session_lock:
            return list(self.active_sessions)

    def get_all_sessions(self) -> Dict[str, SessionInfo]:
        """Get all sessions and their information.

        Returns:
            Dictionary mapping session IDs to SessionInfo objects
        """
        with self.session_lock:
            return self.sessions.copy()

    def register_session_callback(self, session_id: str, callback: callable) -> bool:
        """Register a callback for session events.

        Args:
            session_id: Session identifier
            callback: Callback function that accepts (event_type, session_info)

        Returns:
            True if callback was registered successfully, False otherwise
        """
        with self.session_lock:
            if session_id not in self.sessions:
                self.logger.error(f"Session {session_id} not found")
                return False
                
            self.session_callbacks[session_id].append(callback)
            self.logger.debug(f"Registered callback for session {session_id}")
            return True

    def unregister_session_callback(self, session_id: str, callback: callable) -> bool:
        """Unregister a session callback.

        Args:
            session_id: Session identifier
            callback: Callback function to remove

        Returns:
            True if callback was removed successfully, False otherwise
        """
        with self.session_lock:
            if session_id not in self.session_callbacks:
                return False
                
            try:
                self.session_callbacks[session_id].remove(callback)
                self.logger.debug(f"Unregistered callback for session {session_id}")
                return True
            except ValueError:
                return False

    def update_session_state(self, session_id: str, new_state: SessionState, error_msg: Optional[str] = None) -> None:
        """Update the state of a session.

        Args:
            session_id: Session identifier
            new_state: New session state
            error_msg: Optional error message
        """
        with self.session_lock:
            if session_id not in self.sessions:
                self.logger.warning(f"Attempted to update unknown session {session_id}")
                return
                
            session_info = self.sessions[session_id]
            old_state = session_info.state
            session_info.state = new_state
            
            # Update additional fields based on state
            if new_state == SessionState.CONNECTED:
                session_info.connected_at = time.time()
                
            elif new_state == SessionState.LOGGED_ON:
                self.active_sessions.add(session_id)
                session_info.last_heartbeat = time.time()
                
            elif new_state in [SessionState.LOGGED_OUT, SessionState.DISCONNECTED]:
                self.active_sessions.discard(session_id)
                
            elif new_state == SessionState.ERROR:
                session_info.error_count += 1
                session_info.last_error = error_msg
                self.active_sessions.discard(session_id)
                
            self.logger.info(f"Session {session_id} state changed: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            event_type = new_state.value
            self._notify_session_callbacks(session_id, event_type, session_info)

    def update_message_stats(self, session_id: str, sent: int = 0, received: int = 0) -> None:
        """Update message statistics for a session.

        Args:
            session_id: Session identifier
            sent: Number of messages sent (increment)
            received: Number of messages received (increment)
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return
                
            session_info = self.sessions[session_id]
            session_info.message_count_sent += sent
            session_info.message_count_received += received
            
            # Update last heartbeat time if session is active
            if session_id in self.active_sessions:
                session_info.last_heartbeat = time.time()

    def start_monitoring(self) -> None:
        """Start the session monitoring thread."""
        if self.monitoring_active:
            self.logger.warning("Session monitoring is already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_sessions,
            name="SessionMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Started session monitoring")

    def stop_monitoring(self) -> None:
        """Stop the session monitoring thread."""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped session monitoring")

    def _monitor_sessions(self) -> None:
        """Monitor sessions for timeouts and health checks."""
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                with self.session_lock:
                    sessions_to_check = list(self.sessions.items())
                    
                for session_id, session_info in sessions_to_check:
                    # Check for session timeout
                    if (session_info.last_heartbeat and 
                        current_time - session_info.last_heartbeat > self.session_timeout):
                        
                        self.logger.warning(f"Session {session_id} timed out")
                        self.update_session_state(session_id, SessionState.ERROR, "Session timeout")
                        
                    # Check for stale connecting sessions
                    elif (session_info.state == SessionState.CONNECTING and
                          current_time - session_info.created_at > 60):  # 60 second connection timeout
                        
                        self.logger.warning(f"Session {session_id} connection timeout")
                        self.update_session_state(session_id, SessionState.ERROR, "Connection timeout")
                        
                # Sleep for monitoring interval
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in session monitoring: {e}")
                time.sleep(5)  # Sleep briefly before retrying

    def _notify_session_callbacks(self, session_id: str, event_type: str, session_info: SessionInfo) -> None:
        """Notify all callbacks for a session event.

        Args:
            session_id: Session identifier
            event_type: Type of event
            session_info: Session information
        """
        callbacks = self.session_callbacks.get(session_id, [])
        for callback in callbacks:
            try:
                callback(event_type, session_info)
            except Exception as e:
                self.logger.error(f"Error in session callback: {e}")

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session statistics.

        Returns:
            Dictionary of session statistics
        """
        with self.session_lock:
            total_sessions = len(self.sessions)
            active_sessions = len(self.active_sessions)
            
            state_counts = {}
            total_messages_sent = 0
            total_messages_received = 0
            total_errors = 0
            
            for session_info in self.sessions.values():
                state = session_info.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
                total_messages_sent += session_info.message_count_sent
                total_messages_received += session_info.message_count_received
                total_errors += session_info.error_count
                
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "state_distribution": state_counts,
                "total_messages_sent": total_messages_sent,
                "total_messages_received": total_messages_received,
                "total_errors": total_errors,
            }

    def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive sessions.

        Args:
            max_age_hours: Maximum age in hours for inactive sessions

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        sessions_to_remove = []
        
        with self.session_lock:
            for session_id, session_info in self.sessions.items():
                # Only remove inactive sessions that are old
                if (session_info.state in [SessionState.DISCONNECTED, SessionState.ERROR] and
                    current_time - session_info.created_at > max_age_seconds):
                    sessions_to_remove.append(session_id)
                    
        # Remove the sessions
        for session_id in sessions_to_remove:
            self.remove_session(session_id)
            
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
            
        return len(sessions_to_remove)