#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import datetime
import queue
import quickfix as fix
import quickfix44 as fix44
from argparse import ArgumentParser
from threading import Event, Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("FIXClient")

class FIXClient(fix.Application):
    """
    FIX Client Application class that handles FIX protocol messages and events.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("FIXClient.Application")
        self.session_id = None
        self.is_logged_on = False
        self.message_queue = queue.Queue()
        self.message_store = {}
        self.orders = {}
        self.response_event = Event()
        self.response_lock = Lock()
        self.last_response = None
        self.expected_responses = {}
        
    def onCreate(self, sessionID):
        """Called when a new FIX session is created."""
        self.logger.info(f"Session created: {sessionID}")
        self.session_id = sessionID
        return
        
    def onLogon(self, sessionID):
        """Called when client logs on to the server."""
        self.logger.info(f"Logged on: {sessionID}")
        self.is_logged_on = True
        return
        
    def onLogout(self, sessionID):
        """Called when client logs off from the server."""
        self.logger.info(f"Logged off: {sessionID}")
        self.is_logged_on = False
        return
        
    def toAdmin(self, message, sessionID):
        """Called before an admin message is sent out."""
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)
        self.logger.debug(f"Admin message sent: {msgType} to {sessionID}")
        return
        
    def fromAdmin(self, message, sessionID):
        """Called when an admin message is received."""
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)
        self.logger.debug(f"Admin message received: {msgType} from {sessionID}")
        return
        
    def toApp(self, message, sessionID):
        """Called before an application message is sent out."""
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)
        self.logger.info(f"Sending application message: {msgType} to {sessionID}")
        self.logger.debug(f"Message content: {message}")
        return
        
    def fromApp(self, message, sessionID):
        """
        Called when an application message is received.
        This is where we handle various FIX message types.
        """
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)
        self.logger.info(f"Received application message: {msgType} from {sessionID}")
        self.logger.debug(f"Message content: {message}")
        
        # Store the message for later processing
        self.message_queue.put(message)
        
        # Process the message based on its type
        if msgType.getValue() == fix.MsgType_ExecutionReport:
            self.handle_execution_report(message)
        elif msgType.getValue() == fix.MsgType_OrderCancelReject:
            self.handle_order_cancel_reject(message)
        elif msgType.getValue() == fix.MsgType_MarketDataSnapshotFullRefresh:
            self.handle_market_data_snapshot(message)
        else:
            self.logger.warning(f"Unhandled message type: {msgType.getValue()}")
        
        # Notify any waiting threads that a response has arrived
        with self.response_lock:
            self.last_response = message
            self.response_event.set()
        
        return
    
    def handle_execution_report(self, message):
        """Handle an execution report message."""
        cl_ord_id = fix.ClOrdID()
        message.getField(cl_ord_id)
        
        order_status = fix.OrdStatus()
        message.getField(order_status)
        
        # Store the execution report in our order dictionary
        if cl_ord_id.getValue() not in self.orders:
            self.orders[cl_ord_id.getValue()] = []
        
        self.orders[cl_ord_id.getValue()].append(self.message_to_dict(message))
        
        # Check if this matches an expected response
        if cl_ord_id.getValue() in self.expected_responses:
            expected = self.expected_responses[cl_ord_id.getValue()]
            if expected["msg_type"] == fix.MsgType_ExecutionReport:
                self.logger.info(f"Received expected execution report for order {cl_ord_id.getValue()}")
    
    def handle_order_cancel_reject(self, message):
        """Handle an order cancel reject message."""
        cl_ord_id = fix.ClOrdID()
        message.getField(cl_ord_id)
        
        # Store the cancel reject
        if cl_ord_id.getValue() not in self.orders:
            self.orders[cl_ord_id.getValue()] = []
        
        self.orders[cl_ord_id.getValue()].append(self.message_to_dict(message))
    
    def handle_market_data_snapshot(self, message):
        """Handle a market data snapshot message."""
        md_req_id = fix.MDReqID()
        message.getField(md_req_id)
        
        # Store the market data snapshot
        if md_req_id.getValue() not in self.message_store:
            self.message_store[md_req_id.getValue()] = []
        
        self.message_store[md_req_id.getValue()].append(self.message_to_dict(message))
    
    def message_to_dict(self, message):
        """Convert a FIX message to a dictionary for easier handling."""
        result = {}
        
        # Get message type
        header = message.getHeader()
        msg_type = fix.MsgType()
        header.getField(msg_type)
        result["MsgType"] = msg_type.getValue()
        
        # Get all fields
        for i in range(message.getNumFields()):
            field = fix.FieldMap.getField(message, i+1)
            field_tag = field.getTag()
            field_value = field.getValue()
            result[field_tag] = field_value
        
        # Handle repeating groups if needed
        # This is a simplified approach - full implementation would need to handle all group types
        if result["MsgType"] == fix.MsgType_MarketDataSnapshotFullRefresh:
            if fix.NoMDEntries().getTag() in result:
                no_md_entries = int(result[fix.NoMDEntries().getTag()])
                result["MDEntries"] = []
                
                for i in range(no_md_entries):
                    group = fix44.MarketDataSnapshotFullRefresh().NoMDEntries()
                    message.getGroup(i+1, group)
                    entry = {}
                    
                    # Extract group fields
                    md_entry_type = fix.MDEntryType()
                    group.getField(md_entry_type)
                    entry["MDEntryType"] = md_entry_type.getValue()
                    
                    md_entry_px = fix.MDEntryPx()
                    group.getField(md_entry_px)
                    entry["MDEntryPx"] = md_entry_px.getValue()
                    
                    md_entry_size = fix.MDEntrySize()
                    if group.isSetField(md_entry_size):
                        group.getField(md_entry_size)
                        entry["MDEntrySize"] = md_entry_size.getValue()
                    
                    result["MDEntries"].append(entry)
        
        return result
    
    def wait_for_response(self, timeout=10):
        """Wait for a response message with timeout."""
        self.response_event.clear()
        if self.response_event.wait(timeout):
            with self.response_lock:
                response = self.last_response
                self.last_response = None
                return response
        return None
    
    def send_new_order_single(self, cl_ord_id, symbol, side, order_qty, ord_type, price=None, tif="0"):
        """
        Send a new order single message.
        
        Args:
            cl_ord_id (str): Client Order ID
            symbol (str): Trading symbol (e.g., 'AAPL')
            side (str): Side ('1' for Buy, '2' for Sell)
            order_qty (float): Order quantity
            ord_type (str): Order type ('1' for Market, '2' for Limit)
            price (float, optional): Price (required for Limit orders)
            tif (str, optional): Time In Force ('0' for Day, '1' for GTC)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_logged_on:
            self.logger.error("Not logged on to server")
            return False
        
        # Create a New Order Single message
        new_order = fix44.NewOrderSingle()
        
        # Set required fields
        new_order.setField(fix.ClOrdID(cl_ord_id))
        new_order.setField(fix.Symbol(symbol))
        new_order.setField(fix.Side(side))
        new_order.setField(fix.OrderQty(float(order_qty)))
        new_order.setField(fix.OrdType(ord_type))
        
        # Set optional fields
        if ord_type == fix.OrdType_LIMIT and price is not None:
            new_order.setField(fix.Price(float(price)))
        
        new_order.setField(fix.TimeInForce(tif))
        new_order.setField(fix.TransactTime())
        new_order.setField(fix.HandlInst('1'))  # Manual
        
        # Set up an expected response
        self.expected_responses[cl_ord_id] = {
            "msg_type": fix.MsgType_ExecutionReport,
            "sent_time": datetime.datetime.now()
        }
        
        # Send the message
        try:
            fix.Session.sendToTarget(new_order, self.session_id)
            return True
        except fix.RuntimeError as e:
            self.logger.error(f"Failed to send new order: {e}")
            return False
    
    def send_order_cancel_request(self, cl_ord_id, orig_cl_ord_id, symbol, side, order_qty):
        """
        Send an order cancel request.
        
        Args:
            cl_ord_id (str): New Client Order ID for the cancel request
            orig_cl_ord_id (str): Original Client Order ID of the order to cancel
            symbol (str): Trading symbol
            side (str): Side ('1' for Buy, '2' for Sell)
            order_qty (float): Order quantity
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_logged_on:
            self.logger.error("Not logged on to server")
            return False
        
        # Create an Order Cancel Request message
        cancel_request = fix44.OrderCancelRequest()
        
        # Set required fields
        cancel_request.setField(fix.ClOrdID(cl_ord_id))
        cancel_request.setField(fix.OrigClOrdID(orig_cl_ord_id))
        cancel_request.setField(fix.Symbol(symbol))
        cancel_request.setField(fix.Side(side))
        cancel_request.setField(fix.TransactTime())
        cancel_request.setField(fix.OrderQty(float(order_qty)))
        
        # Set up an expected response
        self.expected_responses[cl_ord_id] = {
            "msg_type": fix.MsgType_ExecutionReport,  # Expect an Execution Report for the cancel
            "sent_time": datetime.datetime.now()
        }
        
        # Send the message
        try:
            fix.Session.sendToTarget(cancel_request, self.session_id)
            return True
        except fix.RuntimeError as e:
            self.logger.error(f"Failed to send cancel request: {e}")
            return False
    
    def send_market_data_request(self, md_req_id, symbol, subscription_type="1", market_depth=0):
        """
        Send a market data request.
        
        Args:
            md_req_id (str): Market Data Request ID
            symbol (str): Trading symbol to get data for
            subscription_type (str): Subscription type ('0' for Snapshot, '1' for Snapshot+Updates)
            market_depth (int): Market depth (0 for full book)
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_logged_on:
            self.logger.error("Not logged on to server")
            return False
        
        # Create a Market Data Request message
        md_request = fix44.MarketDataRequest()
        
        # Set required fields
        md_request.setField(fix.MDReqID(md_req_id))
        md_request.setField(fix.SubscriptionRequestType(subscription_type))
        md_request.setField(fix.MarketDepth(market_depth))
        
        # Set repeating group for symbols
        no_related_sym = fix.NoRelatedSym(1)
        md_request.setField(no_related_sym)
        
        group = fix44.MarketDataRequest().NoRelatedSym()
        group.setField(fix.Symbol(symbol))
        md_request.addGroup(group)
        
        # Set up an expected response
        self.expected_responses[md_req_id] = {
            "msg_type": fix.MsgType_MarketDataSnapshotFullRefresh,
            "sent_time": datetime.datetime.now()
        }
        
        # Send the message
        try:
            fix.Session.sendToTarget(md_request, self.session_id)
            return True
        except fix.RuntimeError as e:
            self.logger.error(f"Failed to send market data request: {e}")
            return False
    
    def get_order_status(self, cl_ord_id):
        """Get the current status of an order."""
        if cl_ord_id in self.orders and self.orders[cl_ord_id]:
            # Return the most recent execution report
            latest_report = self.orders[cl_ord_id][-1]
            return latest_report
        return None
    
    def get_market_data(self, md_req_id):
        """Get the most recent market data for a request."""
        if md_req_id in self.message_store and self.message_store[md_req_id]:
            # Return the most recent market data snapshot
            latest_data = self.message_store[md_req_id][-1]
            return latest_data
        return None


def main():
    """Main function to run the FIX client."""
    parser = ArgumentParser(description="FIX Client for testing")
    parser.add_argument("-c", "--config", default="config/client_config.cfg",
                        help="Path to the FIX client configuration file")
    args = parser.parse_args()
    
    try:
        # Create the FIX application
        app = FIXClient()
        
        # Create the FIX settings
        settings = fix.SessionSettings(args.config)
        
        # Create the FIX store factory and log factory
        store_factory = fix.FileStoreFactory(settings)
        log_factory = fix.FileLogFactory(settings)
        
        # Create the FIX initiator (client)
        initiator = fix.SocketInitiator(app, store_factory, settings, log_factory)
        
        # Start the initiator
        initiator.start()
        logger.info("FIX Client started. Press Ctrl+C to quit.")
        
        # Wait for logon
        timeout = 10
        start_time = time.time()
        while not app.is_logged_on and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not app.is_logged_on:
            logger.error("Failed to log on to server")
            initiator.stop()
            return 1
        
        logger.info("Logged on to server. Ready to send messages.")
        
        # Example: Send a New Order Single
        order_id = f"ORD{int(time.time())}"
        app.send_new_order_single(
            order_id, "AAPL", fix.Side_BUY, 100, fix.OrdType_MARKET
        )
        
        # Wait for a response
        time.sleep(2)
        
        # Check order status
        status = app.get_order_status(order_id)
        if status:
            logger.info(f"Order status: {status}")
        
        # Keep the client running until interrupted
        while True:
            time.sleep(1)
            
    except fix.ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except fix.RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.info("Shutting down FIX Client...")
        if 'initiator' in locals():
            initiator.stop()
        return 0


if __name__ == "__main__":
    sys.exit(main())