"""
Command-line interface for the FixTester framework.
"""
import json
import os
import sys
import time
from typing import Dict, List, Optional

import click
import yaml

from fixtester.core.engine import FixEngine
from fixtester.core.message_factory import MessageFactory
from fixtester.utils.config_loader import ConfigLoader


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """FixTester - Automated FIX Protocol Testing Framework."""
    pass


@cli.command("start-engine")
@click.option("--config", "-c", default="config/default.yaml", help="Path to configuration file")
def start_engine(config):
    """Start the FIX engine with the given configuration."""
    try:
        engine = FixEngine.from_config(config)
        engine.start()
        
        click.echo(f"Engine started with configuration from {config}")
        click.echo("Press Ctrl+C to stop...")
        
        # Keep the engine running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("Stopping engine...")
            engine.stop()
            click.echo("Engine stopped.")
    except Exception as e:
        click.echo(f"Error starting engine: {e}", err=True)
        sys.exit(1)


@cli.command("run-scenario")
@click.argument("scenario_file")
@click.option("--config", "-c", default="config/default.yaml", help="Path to configuration file")
@click.option("--output", "-o", help="Output file for results")
@click.option("--timeout", "-t", default=30, help="Timeout in seconds for the scenario")
def run_scenario(scenario_file, config, output, timeout):
    """Run a test scenario from a scenario file."""
    try:
        # Load the scenario
        try:
            with open(scenario_file, 'r') as f:
                scenario = yaml.safe_load(f)
        except Exception as e:
            click.echo(f"Error loading scenario file: {e}", err=True)
            sys.exit(1)
            
        click.echo(f"Running scenario: {scenario.get('name', 'Unnamed')}")
        
        # Initialize the engine
        engine = FixEngine.from_config(config)
        engine.start()
        
        try:
            # Wait for session establishment
            if not engine.application.sessions:
                click.echo("Waiting for session establishment...")
                if hasattr(engine, "wait_for_logon") and callable(engine.wait_for_logon):
                    if not engine.wait_for_logon(timeout=10.0):
                        click.echo("Timed out waiting for session establishment.", err=True)
                        sys.exit(1)
                else:
                    # Sleep a bit to allow session establishment
                    time.sleep(5)
            
            # Create message factory
            factory = MessageFactory(
                protocol_version=scenario.get("protocol_version", "FIX.4.4")
            )
            
            # Process scenario steps
            results = []
            for i, step in enumerate(scenario.get("steps", [])):
                step_name = step.get("name", f"Step {i+1}")
                step_type = step.get("type", "send")
                
                click.echo(f"Executing step: {step_name} ({step_type})")
                
                if step_type == "send":
                    # Create and send message
                    message_type = step.get("message_type")
                    fields = step.get("fields", {})
                    
                    message = factory.create_message(message_type, fields)
                    
                    # Send and optionally wait for response
                    if step.get("wait_for_response", False):
                        response = engine.send_and_wait(
                            message, 
                            timeout=step.get("timeout", 5.0)
                        )
                        
                        if response:
                            # Validate response if validation rules are provided
                            validation_result = True
                            validation_errors = []
                            if "validation" in step:
                                validation_result, validation_errors = engine.validator.validate_message(
                                    response,
                                    expected_type=step["validation"].get("expected_type"),
                                    expected_fields=step["validation"].get("expected_fields"),
                                    extra_rules=step["validation"].get("extra_rules")
                                )
                                
                            # Store result
                            results.append({
                                "step": step_name,
                                "success": validation_result,
                                "errors": validation_errors,
                                "response": factory.message_to_dict(response)
                            })
                            
                            if not validation_result:
                                click.echo(f"Validation failed: {validation_errors}", err=True)
                                if step.get("fail_on_validation_error", True):
                                    raise Exception(f"Validation failed for step {step_name}")
                        else:
                            click.echo(f"No response received within timeout", err=True)
                            results.append({
                                "step": step_name,
                                "success": False,
                                "errors": ["No response received within timeout"],
                                "response": None
                            })
                            
                            if step.get("fail_on_timeout", True):
                                raise Exception(f"Timeout waiting for response in step {step_name}")
                    else:
                        # Just send without waiting for response
                        success = engine.send_message(message)
                        results.append({
                            "step": step_name,
                            "success": success,
                            "errors": [] if success else ["Failed to send message"],
                            "response": None
                        })
                        
                        if not success and step.get("fail_on_send_error", True):
                            raise Exception(f"Failed to send message in step {step_name}")
                
                elif step_type == "wait":
                    # Just wait for a specified time
                    wait_time = step.get("time", 1.0)
                    click.echo(f"Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                    results.append({
                        "step": step_name,
                        "success": True,
                        "errors": [],
                        "response": None
                    })
                
                elif step_type == "assert":
                    # Assert some condition
                    # TODO: Implement more assertion types
                    click.echo("Assert step not fully implemented yet")
                    results.append({
                        "step": step_name,
                        "success": True,
                        "errors": ["Assert step not fully implemented"],
                        "response": None
                    })
                
                else:
                    click.echo(f"Unknown step type: {step_type}", err=True)
                    results.append({
                        "step": step_name,
                        "success": False,
                        "errors": [f"Unknown step type: {step_type}"],
                        "response": None
                    })
            
            # Prepare scenario results
            scenario_results = {
                "name": scenario.get("name", "Unnamed"),
                "timestamp": time.time(),
                "success": all(step["success"] for step in results),
                "steps": results
            }
            
            # Output results
            if output:
                with open(output, 'w') as f:
                    json.dump(scenario_results, f, indent=2)
                click.echo(f"Results written to {output}")
            
            # Print summary
            click.echo("\nScenario Results:")
            click.echo(f"  Name: {scenario.get('name', 'Unnamed')}")
            click.echo(f"  Success: {scenario_results['success']}")
            click.echo(f"  Steps: {len(results)}")
            click.echo(f"  Passed: {sum(1 for step in results if step['success'])}")
            click.echo(f"  Failed: {sum(1 for step in results if not step['success'])}")
            
            if not scenario_results['success']:
                sys.exit(1)
                
        finally:
            # Stop the engine
            engine.stop()
            
    except Exception as e:
        click.echo(f"Error running scenario: {e}", err=True)
        sys.exit(1)


@cli.command("run-all-scenarios")
@click.option("--scenarios-dir", "-d", default="config/scenarios", help="Directory containing scenario files")
@click.option("--config", "-c", default="config/default.yaml", help="Path to configuration file")
@click.option("--output-dir", "-o", default="results", help="Output directory for results")
@click.option("--fail-fast", "-f", is_flag=True, help="Stop after first failing scenario")
def run_all_scenarios(scenarios_dir, config, output_dir, fail_fast):
    """Run all test scenarios in the specified directory."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all scenario files
        scenario_files = []
        for file in os.listdir(scenarios_dir):
            if file.endswith((".yaml", ".yml")):
                scenario_files.append(os.path.join(scenarios_dir, file))
                
        if not scenario_files:
            click.echo(f"No scenario files found in {scenarios_dir}", err=True)
            sys.exit(1)
            
        click.echo(f"Found {len(scenario_files)} scenario files")
        
        # Run each scenario
        results = []
        failed = False
        
        for scenario_file in scenario_files:
            filename = os.path.basename(scenario_file)
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.json")
            
            click.echo(f"\nRunning scenario file: {filename}")
            
            # Run the scenario via the run-scenario command
            exit_code = os.system(
                f"{sys.executable} -m fixtester.cli run-scenario "
                f"{scenario_file} --config {config} --output {output_file}"
            )
            
            success = exit_code == 0
            results.append({
                "file": filename,
                "success": success
            })
            
            if not success:
                failed = True
                if fail_fast:
                    click.echo("Stopping due to scenario failure (--fail-fast enabled)")
                    break
        
        # Print overall summary
        click.echo("\nOverall Results:")
        click.echo(f"  Total scenarios: {len(results)}")
        click.echo(f"  Passed: {sum(1 for r in results if r['success'])}")
        click.echo(f"  Failed: {sum(1 for r in results if not r['success'])}")
        
        if failed:
            click.echo("\nFailed scenarios:")
            for result in results:
                if not result["success"]:
                    click.echo(f"  - {result['file']}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error running scenarios: {e}", err=True)
        sys.exit(1)


@cli.command("create-scenario")
@click.argument("name")
@click.option("--output", "-o", help="Output file for the new scenario")
def create_scenario(name, output):
    """Create a new scenario template."""
    # Create a basic scenario template
    scenario = {
        "name": name,
        "description": "A test scenario for FIX messages",
        "protocol_version": "FIX.4.4",
        "steps": [
            {
                "name": "Login",
                "type": "send",
                "message_type": "Logon",
                "fields": {
                    "HeartBtInt": "30",
                    "EncryptMethod": "0"
                },
                "wait_for_response": True,
                "timeout": 5.0,
                "fail_on_timeout": True
            },
            {
                "name": "Send Order",
                "type": "send",
                "message_type": "NewOrderSingle",
                "fields": {
                    "ClOrdID": "TEST-ORDER-1",
                    "Symbol": "AAPL",
                    "Side": "1",
                    "TransactTime": "",  # Will be filled automatically
                    "OrderQty": "100",
                    "OrdType": "2",  # Limit order
                    "Price": "150.50",
                    "TimeInForce": "0"  # Day
                },
                "wait_for_response": True,
                "timeout": 5.0,
                "validation": {
                    "expected_type": "ExecutionReport",
                    "expected_fields": {
                        "OrdStatus": "0"  # New
                    }
                }
            },
            {
                "name": "Wait",
                "type": "wait",
                "time": 1.0
            },
            {
                "name": "Cancel Order",
                "type": "send",
                "message_type": "OrderCancelRequest",
                "fields": {
                    "ClOrdID": "TEST-CANCEL-1",
                    "OrigClOrdID": "TEST-ORDER-1",
                    "Symbol": "AAPL",
                    "Side": "1",
                    "TransactTime": ""  # Will be filled automatically
                },
                "wait_for_response": True,
                "timeout": 5.0,
                "validation": {
                    "expected_type": "ExecutionReport",
                    "expected_fields": {
                        "ExecType": "4",  # Canceled
                        "OrdStatus": "4"  # Canceled
                    }
                }
            },
            {
                "name": "Logout",
                "type": "send",
                "message_type": "Logout",
                "fields": {},
                "wait_for_response": True,
                "timeout": 5.0
            }
        ]
    }
    
    # Determine output file if not provided
    if not output:
        safe_name = name.lower().replace(" ", "_")
        output = f"config/scenarios/{safe_name}.yaml"
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Write scenario to file
    with open(output, 'w') as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
        
    click.echo(f"Created scenario template: {output}")


@cli.command("list-scenarios")
@click.option("--dir", "-d", default="config/scenarios", help="Directory containing scenario files")
def list_scenarios(dir):
    """List available test scenarios."""
    try:
        if not os.path.exists(dir):
            click.echo(f"Scenarios directory not found: {dir}", err=True)
            sys.exit(1)
            
        scenario_files = []
        for file in os.listdir(dir):
            if file.endswith((".yaml", ".yml")):
                scenario_files.append(file)
                
        if not scenario_files:
            click.echo(f"No scenario files found in {dir}")
            return
            
        click.echo(f"Found {len(scenario_files)} scenario files:")
        
        for file in sorted(scenario_files):
            try:
                with open(os.path.join(dir, file), 'r') as f:
                    scenario = yaml.safe_load(f)
                    name = scenario.get("name", "Unnamed")
                    description = scenario.get("description", "No description")
                    step_count = len(scenario.get("steps", []))
                    protocol = scenario.get("protocol_version", "Unknown")
                    
                    click.echo(f"  - {file}")
                    click.echo(f"      Name: {name}")
                    click.echo(f"      Description: {description}")
                    click.echo(f"      Steps: {step_count}")
                    click.echo(f"      Protocol: {protocol}")
                    click.echo("")
            except Exception as e:
                click.echo(f"  - {file} [Error: {e}]")
                
    except Exception as e:
        click.echo(f"Error listing scenarios: {e}", err=True)
        sys.exit(1)


@cli.command("validate-config")
@click.argument("config_file")
def validate_config(config_file):
    """Validate a configuration file."""
    try:
        click.echo(f"Validating configuration file: {config_file}")
        config = ConfigLoader.load(config_file)
        
        # Check required sections
        required_sections = ["fix", "logging"]
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            click.echo(f"Missing required sections: {', '.join(missing_sections)}", err=True)
            sys.exit(1)
            
        # Check fix section
        fix_config = config.get("fix", {})
        fix_required_fields = ["version", "connection"]
        missing_fix_fields = [field for field in fix_required_fields if field not in fix_config]
        
        if missing_fix_fields:
            click.echo(f"Missing required fields in fix section: {', '.join(missing_fix_fields)}", err=True)
            sys.exit(1)
            
        # Check connection section
        connection = fix_config.get("connection", {})
        conn_required_fields = ["host", "port", "sender_comp_id", "target_comp_id"]
        missing_conn_fields = [field for field in conn_required_fields if field not in connection]
        
        if missing_conn_fields:
            click.echo(f"Missing required fields in connection section: {', '.join(missing_conn_fields)}", err=True)
            sys.exit(1)
            
        # All checks passed
        click.echo("Configuration is valid.")
        
    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()