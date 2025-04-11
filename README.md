# FixTester

A lightweight test automation suite for simulating FIX (Financial Information eXchange) protocol messages between clients and a trading server.

## Project Description

FixTester provides a robust framework for testing FIX protocol implementations by simulating both client and server behaviors. Built using the QuickFIX engine, this suite allows developers and QA engineers to:

- Simulate a FIX server accepting client connections
- Send and receive various FIX message types
- Validate message sequence and content
- Run automated test scenarios with pass/fail verification
- Deploy in containerized environments for CI/CD pipelines

The project is designed to be extensible, allowing for custom test scenarios and message validation rules.

## Features

- FIX protocol versions support: FIX.4.2, FIX.4.4, and FIX.5.0
- Bidirectional message testing (client-to-server and server-to-client)
- Automated test scenarios with detailed reporting
- Configurable message validation rules
- Docker support for easy deployment and CI integration
- Comprehensive logging of all FIX messages
