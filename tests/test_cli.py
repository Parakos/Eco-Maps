"""Tests for CLI commands."""

import pytest
import subprocess
import json
from pathlib import Path


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        ['python', '-m', 'eco_maps.cli', '--help'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert 'ecotrain' in result.stdout
    assert 'ecopredict' in result.stdout
    assert 'ecoserve' in result.stdout
    assert 'ecosimulate' in result.stdout


def test_config_exists():
    """Test that default config file exists."""
    config_path = Path('config/default_config.yml')
    assert config_path.exists(), "Default config file should exist"


def test_fixtures_exist():
    """Test that required fixture files exist."""
    fixtures = [
        'data/fixtures/trajectories.json',
        'data/fixtures/candidate_routes.json',
        'data/fixtures/weather_timeseries.json',
        'data/fixtures/grid_carbon.json',
        'data/fixtures/sample_scenario.json'
    ]
    
    for fixture in fixtures:
        assert Path(fixture).exists(), f"Fixture {fixture} should exist"


def test_scenario_json_valid():
    """Test that sample scenario JSON is valid."""
    with open('data/fixtures/sample_scenario.json', 'r') as f:
        scenario = json.load(f)
    
    assert 'initial_position' in scenario
    assert 'events' in scenario
    assert len(scenario['events']) > 0
