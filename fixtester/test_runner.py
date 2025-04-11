#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import logging
import datetime
import json
import argparse
import quickfix as fix
import quickfix44 as fix44
import subprocess
from threading import Thread

from fixtester.client import FIXClient
from fixtester.message_validator import MessageValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("TestRunner")

class TestResult:
    """Stores the result of a test case."""
    
    def __init__(self, scenario_name, status="unknown", message="", duration=0, steps=None):
        self.scenario_name = scenario_name
        self.status = status  # 'pass', 'fail', 'skip', 'unknown'
        self.message = message
        self.duration = duration
        self.steps = steps or []
        self.timestamp = datetime.datetime.now()
    
    def to_dict(self):
        """Convert the test result to a dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "status": self.status,
            "message": self.message,
            "duration": self.duration,
            "steps": self.steps,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self):
        """String representation of the test result."""
        return f"{self.scenario_name}: {self.status} ({self.message})"


class TestRunner:
    """
    Test runner for FIX protocol testing.
    Runs test scenarios against a FIX server.
    """
    
    def __init__(self, client_config, test_scenarios_path):
        self.logger = logging.getLogger("TestRunner")
        self.client_config = client_config
        self.test_scenarios_path = test_scenarios_path
        self.validator = MessageValidator()
        self.results = []
        self.client = None
        self._load_test_scenarios()
        
    def _load_test_scenarios(self):
        """Load test scenarios from YAML file."""
        try:
            with open(self.test_scenarios_path, 'r') as f:
                self.scenarios = yaml.safe_load(f)
            
            # Check if there's a 'scenarios' key
            if 'scenarios' in self.scenarios:
                self.scenarios = self.scenarios['scenarios']
                
            self.logger.info(f"Loaded {len(self.scenarios)} test scenarios")
        except Exception as e:
            self.logger.error(f"Failed to load test scenarios: {e}")
            self.scenarios = []
    
    def _setup_client(self):
        """Set up the FIX client."""
        try:
            # Create the FIX application
            self.client = FIXClient()
            
            # Create the FIX settings
            settings = fix.SessionSettings(self.client_config)
            
            # Create the FIX store factory and log factory
            store_factory = fix.FileStoreFactory(settings)
            log_factory = fix.FileLogFactory(settings)
            
            # Create the FIX initiator (client)
            self.initiator = fix.SocketInitiator(self.client, store_factory, settings, log_factory)
            
            # Start the initiator
            self.initiator.start()
            self.logger.info("FIX Client started")
            
            # Wait for logon
            timeout = 10
            start_time = time.time()
            while not self.client.is_logged_on and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.client.is_logged_on:
                self.logger.error("Failed to log on to server")
                self.initiator.stop()
                return False
                
            self.logger.info("Client logged on to server")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up client: {e}")
            if hasattr(self, 'initiator'):
                self.initiator.stop()
            return False
    
    def _teardown_client(self):
        """Tear down the FIX client."""
        if hasattr(self, 'initiator'):
            self.initiator.stop()
            self.logger.info("FIX Client stopped")
    
    def run_all_tests(self):
        """Run all test scenarios."""
        self.results = []
        
        # Set up client
        if not self._setup_client():
            return self.results
        
        try:
            # Run each scenario
            for scenario in self.scenarios:
                result = self.run_scenario(scenario)
                self.results.append(result)
                
                # Add a small delay between tests
                time.sleep(1)
            
        finally:
            # Tear down client
            self._teardown_client()
        
        return self.results
    
    def run_scenario(self, scenario):
        """Run a single test scenario."""
        scenario_name = scenario.get("name", "Unknown scenario")
        self.logger.info(f"Running scenario: {scenario_name}")
        
        start_time = time.time()
        steps_results = []
        scenario_status = "pass"
        scenario_message = ""
        
        try:
            # Process each step in the scenario
            for i, step in enumerate(scenario.get("steps", [])):
                step_result = self._execute_step(step, i+1)
                steps_results.append(step_result)
                
                # If any step fails, mark the scenario as failed
                if step_result["status"] == "fail":
                    scenario_status = "fail"
                    scenario_message = f"Failed at step {i+1}: {step_result['message']}"
                    break
            
            # Check if scenario matches expected result
            expected_result = scenario.get("expected_result", "pass")
            if expected_result != scenario_status:
                scenario_message = f"Scenario expected result '{expected_result}' does not match actual result '{scenario_status}'"
                if expected_result == "fail" and scenario_status == "pass":
                    # This is a negative test case that unexpectedly passed
                    scenario_status = "fail"
                elif expected_result == "pass" and scenario_status == "fail":
                    # Test case failed as expected
                    scenario_status = "fail"
                
        except Exception as e:
            scenario_status = "fail"
            scenario_message = f"Error executing scenario: {str(e)}"
            self.logger.error(f"Error in scenario '{scenario_name}': {e}", exc_info=True)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Create and return the result
        result = TestResult(
            scenario_name=scenario_name,
            status=scenario_status,
            message=scenario_message,
            duration=duration,
            steps=steps_results
        )
        
        # Log the result
        log_level = logging.INFO if scenario_status == "pass" else logging.ERROR
        self.logger.log(log_level, f"Scenario '{scenario_name}' {scenario_status} in {duration:.2f}s: {scenario_message}")
        
        return result
    
    def _execute_step(self, step, step_number):
        """
        Execute a single test step.
        
        Args:
            step (dict): The step configuration
            step_number (int): Step number for logging
        
        Returns:
            dict: Step result
        """
        step_type = step.get("action", "unknown")
        description = step.get("description", f"Step {step_number}")
        self.logger.info(f"Executing step {step_number}: {step_type} - {description}")
        
        step_result = {
            "step_number": step_number,
            "step_type": step_type,
            "description": description,
            "status": "fail",
            "message": "",
            "details": {}
        }
        
        timeout = step.get("timeout", 10)
        
        try:
            if step_type == "connect":
                # Connection step (already done in setup)
                step_result["status"] = "pass"
                step_result["message"] = "Connected to server"
                
            elif step_type == "send":
                # Send a message
                message_type = step.get("message_type")
                from_entity = step.get("from")
                fields = step.get("fields", {})
                
                if message_type == "D":  # New Order Single
                    cl_ord_id = fields.get("ClOrdID", f"AUTO_{int(time.time())}")
                    symbol = fields.get("Symbol", "AAPL")
                    side = fields.get("Side", "1")  # Default to Buy
                    order_qty = fields.get("OrderQty", 100)
                    ord_type = fields.get("OrdType", "1")  # Default to Market
                    price = fields.get("Price") if "Price" in fields else None
                    tif = fields.get("TimeInForce", "0")  # Default to Day
                    
                    # Replace current_time placeholder
                    if isinstance(fields.get("TransactTime"), str) and "{current_time}" in fields["TransactTime"]:
                        # Not needed for QuickFIX as it sets TransactTime automatically
                        pass
                    
                    # Send the order
                    success = self.client.send_new_order_single(
                        cl_ord_id, symbol, side, order_qty, ord_type, price, tif
                    )
                    
                    if success:
                        step_result["status"] = "pass"
                        step_result["message"] = f"Sent New Order Single: {cl_ord_id}"
                        step_result["details"]["message_sent"] = {
                            "ClOrdID": cl_ord_id,
                            "Symbol": symbol,
                            "Side": side,
                            "OrderQty": order_qty,
                            "OrdType": ord_type
                        }
                    else:
                        step_result["message"] = "Failed to send New Order Single"
                
                elif message_type == "F":  # Order Cancel Request
                    cl_ord_id = fields.get("ClOrdID", f"AUTO_{int(time.time())}")
                    orig_cl_ord_id = fields.get("OrigClOrdID")
                    symbol = fields.get("Symbol", "AAPL")
                    side = fields.get("Side", "1")  # Default to Buy
                    order_qty = fields.get("OrderQty", 100)
                    
                    # Send the cancel request
                    success = self.client.send_order_cancel_request(
                        cl_ord_id, orig_cl_ord_id, symbol, side, order_qty
                    )
                    
                    if success:
                        step_result["status"] = "pass"
                        step_result["message"] = f"Sent Order Cancel Request: {cl_ord_id}"
                        step_result["details"]["message_sent"] = {
                            "ClOrdID": cl_ord_id,
                            "OrigClOrdID": orig_cl_ord_id,
                            "Symbol": symbol,
                            "Side": side
                        }
                    else:
                        step_result["message"] = "Failed to send Order Cancel Request"
                
                elif message_type == "V":  # Market Data Request
                    md_req_id = fields.get("MDReqID", f"MDREQ_{int(time.time())}")
                    symbol = fields.get("Symbol", "AAPL")
                    subscription_type = fields.get("SubscriptionRequestType", "1")
                    market_depth = fields.get("MarketDepth", 0)
                    
                    # Send the market data request
                    success = self.client.send_market_data_request(
                        md_req_id, symbol, subscription_type, market_depth
                    )
                    
                    if success:
                        step_result["status"] = "pass"
                        step_result["message"] = f"Sent Market Data Request: {md_req_id}"
                        step_result["details"]["message_sent"] = {
                            "MDReqID": md_req_id,
                            "Symbol": symbol,
                            "SubscriptionRequestType": subscription_type,
                            "MarketDepth": market_depth
                        }
                    else:
                        step_result["message"] = "Failed to send Market Data Request"
                
                else:
                    step_result["message"] = f"Unsupported message type: {message_type}"
            
            elif step_type == "receive":
                # Wait for and validate a received message
                message_type = step.get("message_type")
                to_entity = step.get("to")
                validation_rules = step.get("validation", [])
                
                # Wait for response
                response = self.client.wait_for_response(timeout)
                
                if response:
                    # Convert response to dict for validation
                    response_dict = self.client.message_to_dict(response)
                    
                    # Validate message type
                    type_valid = self.validator.validate_message_type(response_dict, message_type)
                    
                    if type_valid:
                        # Validate message content
                        valid, message = self.validator.validate_message(response_dict, validation_rules)
                        
                        if valid:
                            step_result["status"] = "pass"
                            step_result["message"] = f"Received and validated {message_type} message"
                            step_result["details"]["message_received"] = response_dict
                        else:
                            step_result["message"] = f"Message validation failed: {message}"
                    else:
                        step_result["message"] = f"Expected message type {message_type}, got {response_dict.get('MsgType', 'unknown')}"
                else:
                    step_result["message"] = f"No message received within timeout ({timeout}s)"
            
            elif step_type == "wait":
                # Wait for a specified time
                wait_time = step.get("time", 1)
                time.sleep(wait_time)
                step_result["status"] = "pass"
                step_result["message"] = f"Waited for {wait_time}s"
            
            else:
                step_result["message"] = f"Unknown step type: {step_type}"
        
        except Exception as e:
            step_result["message"] = f"Error executing step: {str(e)}"
            self.logger.error(f"Error in step {step_number}: {e}", exc_info=True)
        
        # Log step result
        log_level = logging.INFO if step_result["status"] == "pass" else logging.ERROR
        self.logger.log(log_level, f"Step {step_number} {step_result['status']}: {step_result['message']}")
        
        return step_result
    
    def generate_report(self, report_path=None):
        """
        Generate a test report.
        
        Args:
            report_path (str, optional): Path to save the report
        
        Returns:
            dict: Report data
        """
        # Default report path
        if not report_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/test_report_{timestamp}.json"
        
        # Create report data
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed": len([r for r in self.results if r.status == "pass"]),
            "failed": len([r for r in self.results if r.status == "fail"]),
            "skipped": len([r for r in self.results if r.status == "skip"]),
            "duration": sum(r.duration for r in self.results),
            "results": [r.to_dict() for r in self.results]
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Save report to file
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to {report_path}")
        return report


def main():
    """Main function to run the test suite."""
    parser = argparse.ArgumentParser(description="FIX Protocol Test Runner")
    parser.add_argument("-c", "--client-config", default="config/client_config.cfg",
                        help="Path to the FIX client configuration file")
    parser.add_argument("-s", "--scenarios", default="config/test_scenarios.yaml",
                        help="Path to the test scenarios YAML file")
    parser.add_argument("-r", "--report", default=None,
                        help="Path to save the test report (default: reports/test_report_TIMESTAMP.json)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run test runner
    runner = TestRunner(args.client_config, args.scenarios)
    runner.run_all_tests()
    runner.generate_report(args.report)
    
    # Print summary
    passed = len([r for r in runner.results if r.status == "pass"])
    failed = len([r for r in runner.results if r.status == "fail"])
    total = len(runner.results)
    
    print("\nTest Summary:")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for result in runner.results:
            if result.status == "fail":
                print(f"- {result.scenario_name}: {result.message}")
    
    # Return exit code based on test results
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())