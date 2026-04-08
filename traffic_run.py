#!/usr/bin/env python3
"""
Traffic Analyzer Agent — Entry Point
Run this to start the interactive CLI.

Usage:
    python traffic_run.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic_analyzer_agent import main

if __name__ == "__main__":
    main()
