"""
Command-line interface for FixTester providing easy access to testing functionality.
"""
import os
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import click

from fixtester.core.engine import FixEngine
from fixtester.core.message_factory import MessageFactory
from fixtester.core.validator import MessageValidator
from fixtester.utils.config_loader import ConfigLoader
from fixtester.utils.logger import setup_logger


@click.group()
@click.option('--config', '-c', default='config/default.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, config, verbose):
    """FixTester - Automated FIX Protocol Testing Framework"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    log_config = {'level': 'DEBUG' if verbose else 'INFO'}
    ctx.obj['logger'] = setup_logger('fixtester.cli', log_config)


@main.command()
@click.option('--scenario', '-s', required=True, help='Path to scenario configuration file')
@click.option('--timeout', '-t', default=30.0, help='Test timeout in seconds')
@click.option('--output', '-o', help='Output file for test results')
@click.pass_context
def run_scenario(ctx, scenario, timeout, output):
    """Run a specific test scenario"""
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        logger.info(f"Running scenario: {scenario}")
        
        # Load main configuration
        config = ConfigLoader.load(config_path)
        
        # Load scenario configuration
        scenario_config = ConfigLoader.load(scenario)
        
        # Create and start the engine
        engine = FixEngine(config)
        engine.start()
        
        # Wait for connection
        logger.info("Waiting for FIX session to establish...")
        time.sleep(2)
        
        # Run the scenario
        results = _run_scenario_tests(engine, scenario_config, timeout)
        
        # Stop the engine
        engine.stop()
        
        # Output results
        if output:
            _save_results(results, output)
        else:
            _print_results(results)
            
        # Exit with appropriate code
        if results['passed']:
            logger.info("All tests passed!")
            sys.exit(0)
        else:
            logger.error("Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running scenario: {e}")
        sys.exit(1)


@main.command()
@click.option('--scenarios-dir', '-d', default='config/scenarios', help='Directory containing scenario files')
@click.option('--timeout', '-t', default=30.0, help='Test timeout in seconds per scenario')
@click.option('--output', '-o', help='Output file for test results')
@click.option('--parallel', '-p', is_flag=True, help='Run scenarios in parallel')
@click.pass_context
def run_all_scenarios(ctx, scenarios_dir, timeout, output, parallel):
    """Run all test scenarios in a directory"""
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        # Find all scenario files
        scenario_files = list(Path(scenarios_dir).glob('*.yaml'))
        if not scenario_files:
            logger.error(f"No scenario files found in {scenarios_dir}")
            sys.exit(1)
            
        logger.info(f"Found {len(scenario_files)} scenario files")
        
        # Load main configuration
        config = ConfigLoader.load(config_path)
        
        all_results = []
        
        if parallel:
            # TODO: Implement parallel execution
            logger.warning("Parallel execution not yet implemented, running sequentially")
        
        # Run scenarios sequentially
        for scenario_file in scenario_files:
            logger.info(f"Running scenario: {scenario_file.name}")
            
            try:
                # Load scenario configuration
                scenario_config = ConfigLoader.load(str(scenario_file))
                
                # Create and start the engine
                engine = FixEngine(config)
                engine.start()
                
                # Wait for connection
                time.sleep(2)
                
                # Run the scenario
                results = _run_scenario_tests(engine, scenario_config, timeout)
                results['scenario_file'] = str(scenario_file)
                all_results.append(results)
                
                # Stop the engine
                engine.stop()
                
                # Brief pause between scenarios
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error running scenario {scenario_file.name}: {e}")
                all_results.append({
                    'scenario_file': str(scenario_file),
                    'passed': False,
                    'error': str(e),
                    'tests': []
                })
        
        # Output combined results
        combined_results = _combine_results(all_results)
        
        if output:
            _save_results(combined_results, output)
        else:
            _print_combined_results(combined_results)
            
        # Exit with appropriate code
        if combined_results['all_passed']:
            logger.info("All scenarios passed!")
            sys.exit(0)
        else:
            logger.error("Some scenarios failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running scenarios: {e}")
        sys.exit(1)


@main.command()
@click.option('--message-type', '-t', required=True, help='FIX message type (e.g., NewOrderSingle)')
@click.option('--fields', '-f', help='JSON string of field values')
@click.option('--template', help='Template name to use')
@click.option('--validate', '-v', is_flag=True, help='Validate the created message')
@click.pass_context
def create_message(ctx, message_type, fields, template, validate):
    """Create a FIX message and optionally validate it"""
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        config = ConfigLoader.load(config_path)
        
        # Create message factory
        factory = MessageFactory()
        
        # Parse fields if provided
        field_values = {}
        if fields:
            field_values = json.loads(fields)
        
        # Create message
        if template:
            message = factory.create_from_template(template, field_values)
        else:
            message = factory.create_message(message_type, field_values)
        
        # Print the message
        click.echo("Created FIX message:")
        click.echo(str(message))
        
        # Validate if requested
        if validate:
            validator = MessageValidator(config.get('validation', {}))
            report = validator.validate_message(message, message_type)
            
            click.echo("\nValidation Report:")
            click.echo(f"Valid: {report.is_valid}")
            click.echo(f"Errors: {report.errors_count}")
            click.echo(f"Warnings: {report.warnings_count}")
            
            if report.issues:
                click.echo("\nIssues:")
                for issue in report.issues:
                    click.echo(f"  {issue.level.value.upper()}: {issue.message}")
        
    except Exception as e:
        logger.error(f"Error creating message: {e}")
        sys.exit(1)


@main.command()
@click.option('--message', '-m', required=True, help='FIX message string to validate')
@click.option('--expected-type', help='Expected message type')
@click.pass_context
def validate_message(ctx, message, expected_type):
    """Validate a FIX message string"""
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        config = ConfigLoader.load(config_path)
        
        # Parse the message string
        # TODO: Implement FIX message parsing from string
        click.echo("Message validation from string not yet implemented")
        
    except Exception as e:
        logger.error(f"Error validating message: {e}")
        sys.exit(1)


@main.command()
@click.option('--host', default='localhost', help='FIX server host')
@click.option('--port', default=9878, help='FIX server port')
@click.option('--sender-comp-id', default='CLIENT', help='Sender CompID')
@click.option('--target-comp-id', default='SERVER', help='Target CompID')
@click.option('--duration', default=60, help='Test duration in seconds')
@click.pass_context
def stress_test(ctx, host, port, sender_comp_id, target_comp_id, duration):
    """Run stress test against a FIX server"""
    logger = ctx.obj['logger']
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        config = ConfigLoader.load(config_path)
        
        # Override connection settings
        config['fix']['connection']['host'] = host
        config['fix']['connection']['port'] = port
        config['fix']['connection']['sender_comp_id'] = sender_comp_id
        config['fix']['connection']['target_comp_id'] = target_comp_id
        
        logger.info(f"Starting stress test against {host}:{port} for {duration} seconds")
        
        # Create and start the engine
        engine = FixEngine(config)
        engine.start()
        
        # Wait for connection
        time.sleep(2)
        
        # Run stress test
        _run_stress_test(engine, duration)
        
        # Stop the engine
        engine.stop()
        
        logger.info("Stress test completed")
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        sys.exit(1)


@main.command()
@click.pass_context
def list_templates(ctx):
    """List available message templates"""
    try:
        factory = MessageFactory()
        templates = factory.get_available_templates()
        
        if templates:
            click.echo("Available templates:")
            for template in templates:
                click.echo(f"  - {template}")
        else:
            click.echo("No templates found")
            
    except Exception as e:
        click.echo(f"Error listing templates: {e}")
        sys.exit(1)


@main.command()
@click.pass_context
def list_message_types(ctx):
    """List supported FIX message types"""
    try:
        factory = MessageFactory()
        message_types = factory.get_supported_message_types()
        
        click.echo("Supported message types:")
        for msg_type in message_types:
            click.echo(f"  - {msg_type}")
            
    except Exception as e:
        click.echo(f"Error listing message types: {e}")
        sys.exit(1)


def _run_scenario_tests(engine: FixEngine, scenario_config: Dict, timeout: float) -> Dict:
    """Run tests defined in a scenario configuration"""
    results = {
        'passed': True,
        'tests': [],
        'start_time': time.time(),
        'end_time': None
    }
    
    tests = scenario_config.get('tests', [])
    factory = MessageFactory()
    
    for i, test in enumerate(tests):
        test_name = test.get('name', f'Test {i+1}')
        test_result = {
            'name': test_name,
            'passed': False,
            'error': None,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            # Create the test message
            message_type = test['message_type']
            fields = test.get('fields', {})
            message = factory.create_message(message_type, fields)
            
            # Send the message and wait for response
            response = engine.send_and_wait(message, timeout)
            
            if response:
                # Validate response if validation rules are provided
                validation_rules = test.get('validation', {})
                if validation_rules:
                    validator = MessageValidator()
                    report = validator.validate_message(response)
                    test_result['passed'] = report.is_valid
                    if not report.is_valid:
                        test_result['error'] = f"Validation failed: {report.issues[0].message if report.issues else 'Unknown error'}"
                else:
                    test_result['passed'] = True
            else:
                test_result['error'] = "No response received"
                
        except Exception as e:
            test_result['error'] = str(e)
        
        test_result['duration'] = time.time() - start_time
        results['tests'].append(test_result)
        
        if not test_result['passed']:
            results['passed'] = False
    
    results['end_time'] = time.time()
    return results


def _run_stress_test(engine: FixEngine, duration: int) -> None:
    """Run a stress test for the specified duration"""
    factory = MessageFactory()
    end_time = time.time() + duration
    message_count = 0
    
    while time.time() < end_time:
        try:
            # Create a new order single message
            message = factory.create_new_order_single(
                cl_ord_id=f"stress_{message_count}",
                symbol="TEST",
                side="1",
                order_qty=100,
                ord_type="2",
                price=100.0
            )
            
            # Send the message (don't wait for response in stress test)
            engine.send_message(message)
            message_count += 1
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error sending message {message_count}: {e}")
    
    print(f"Sent {message_count} messages in {duration} seconds")
    print(f"Average rate: {message_count / duration:.2f} messages/second")


def _save_results(results: Dict, output_path: str) -> None:
    """Save test results to a file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def _print_results(results: Dict) -> None:
    """Print test results to console"""
    click.echo("\n=== Test Results ===")
    click.echo(f"Overall: {'PASSED' if results['passed'] else 'FAILED'}")
    
    if 'tests' in results:
        click.echo(f"Tests run: {len(results['tests'])}")
        click.echo(f"Passed: {sum(1 for t in results['tests'] if t['passed'])}")
        click.echo(f"Failed: {sum(1 for t in results['tests'] if not t['passed'])}")
        
        for test in results['tests']:
            status = 'PASS' if test['passed'] else 'FAIL'
            click.echo(f"  {test['name']}: {status}")
            if test['error']:
                click.echo(f"    Error: {test['error']}")


def _combine_results(all_results: List[Dict]) -> Dict:
    """Combine results from multiple scenarios"""
    combined = {
        'all_passed': True,
        'scenarios': all_results,
        'total_tests': 0,
        'total_passed': 0,
        'total_failed': 0
    }
    
    for result in all_results:
        if not result.get('passed', False):
            combined['all_passed'] = False
        
        tests = result.get('tests', [])
        combined['total_tests'] += len(tests)
        combined['total_passed'] += sum(1 for t in tests if t.get('passed', False))
        combined['total_failed'] += sum(1 for t in tests if not t.get('passed', False))
    
    return combined


def _print_combined_results(results: Dict) -> None:
    """Print combined results from multiple scenarios"""
    click.echo("\n=== Combined Test Results ===")
    click.echo(f"Overall: {'PASSED' if results['all_passed'] else 'FAILED'}")
    click.echo(f"Scenarios run: {len(results['scenarios'])}")
    click.echo(f"Total tests: {results['total_tests']}")
    click.echo(f"Passed: {results['total_passed']}")
    click.echo(f"Failed: {results['total_failed']}")
    
    click.echo("\nScenario Details:")
    for scenario in results['scenarios']:
        status = 'PASS' if scenario.get('passed', False) else 'FAIL'
        click.echo(f"  {Path(scenario['scenario_file']).name}: {status}")
        if scenario.get('error'):
            click.echo(f"    Error: {scenario['error']}")


if __name__ == '__main__':
    main()