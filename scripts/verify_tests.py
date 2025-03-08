#!/usr/bin/env python
"""
Test Verification Script for Credit Cashflow Engine

This script verifies that all tests are passing and generates a test report.
It can be used locally or in CI/CD pipelines to ensure code quality.

Usage:
    python verify_tests.py [--coverage-threshold 80]
"""

import os
import sys
import json
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class TestVerifier:
    """Verifies test status and generates reports."""

    def __init__(self, 
                coverage_threshold: float = 80.0,
                unittest_args: List[str] = None,
                mock_env: bool = False):
        """
        Initialize the test verifier.
        
        Args:
            coverage_threshold: Minimum acceptable code coverage percentage
            unittest_args: Additional arguments to pass to pytest
            mock_env: Whether to mock environment variables for testing
        """
        self.coverage_threshold = coverage_threshold
        self.unittest_args = unittest_args or []
        self.mock_env = mock_env
        
        # Test results
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = 0
        
        # Coverage results
        self.coverage_data = {}
        self.overall_coverage = 0.0
        
        # Test output
        self.test_output = ""
        
        # Base directory
        self.base_dir = Path(__file__).parent.parent.absolute()
    
    def run_tests(self) -> bool:
        """
        Run the tests and collect results.
        
        Returns:
            True if all tests pass and coverage is above threshold, False otherwise
        """
        print("Running tests with pytest...")
        
        # Prepare the command
        cmd = [
            "pytest",
            "--cov=app",
            "--cov-report=xml",
            "--cov-report=term",
            "--junitxml=test-results.xml",
            "-v"
        ]
        
        # Add any additional arguments
        cmd.extend(self.unittest_args)
        
        # Add the tests directory
        cmd.append("tests/")
        
        # Prepare environment variables for testing if needed
        env = os.environ.copy()
        if self.mock_env:
            mock_env_vars = {
                "ENV": "testing",
                "LOG_LEVEL": "debug",
                "WORKERS": "1",
                "CALCULATION_THREAD_POOL_SIZE": "1",
                "SECRET_KEY": "test_secret_key",
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_KEY": "dummy_key",
                "SUPABASE_SERVICE_ROLE_KEY": "dummy_service_role_key",
                "SUPABASE_JWT_SECRET": "dummy_jwt_secret",
                "NEXT_PUBLIC_SUPABASE_URL": "https://example.supabase.co",
                "NEXT_PUBLIC_SUPABASE_ANON_KEY": "dummy_anon_key",
                "UPSTASH_REDIS_HOST": "localhost",
                "UPSTASH_REDIS_PORT": "6379",
                "UPSTASH_REDIS_PASSWORD": "dummy_password",
                # CORS_ORIGINS needs to be a JSON-formatted list as a string
                "CORS_ORIGINS": '["http://localhost:3000", "https://example.com"]'
            }
            env.update(mock_env_vars)
            print("Using mocked environment variables for testing")
        
        # Run the tests
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.base_dir),
                env=env
            )
            
            self.test_output = process.stdout + process.stderr
            print(self.test_output)
            
            # Parse the test results XML
            self._parse_test_results()
            
            # Parse the coverage data
            self._parse_coverage_data()
            
            # Check if all tests passed and coverage is sufficient
            tests_passed = self.failed_tests == 0 and self.errors == 0
            coverage_sufficient = self.overall_coverage >= self.coverage_threshold
            
            return tests_passed and coverage_sufficient
        
        except Exception as e:
            print(f"Error running tests: {e}")
            return False
    
    def _parse_test_results(self) -> None:
        """Parse the JUnit XML test results."""
        try:
            xml_path = self.base_dir / "test-results.xml"
            if not xml_path.exists():
                print("Warning: test-results.xml not found")
                return
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get test counts from the testsuite element
            testsuite = root
            self.total_tests = int(testsuite.attrib.get('tests', 0))
            self.errors = int(testsuite.attrib.get('errors', 0))
            self.failed_tests = int(testsuite.attrib.get('failures', 0))
            self.skipped_tests = int(testsuite.attrib.get('skipped', 0))
            self.passed_tests = self.total_tests - self.failed_tests - self.errors - self.skipped_tests
            
        except Exception as e:
            print(f"Error parsing test results: {e}")
    
    def _parse_coverage_data(self) -> None:
        """Parse the coverage XML data."""
        try:
            xml_path = self.base_dir / "coverage.xml"
            if not xml_path.exists():
                print("Warning: coverage.xml not found")
                return
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get overall coverage
            if 'line-rate' in root.attrib:
                self.overall_coverage = float(root.attrib['line-rate']) * 100
            
            # Get coverage by package
            for package in root.findall('.//package'):
                package_name = package.attrib.get('name', 'unknown')
                line_rate = float(package.attrib.get('line-rate', 0)) * 100
                self.coverage_data[package_name] = line_rate
        
        except Exception as e:
            print(f"Error parsing coverage data: {e}")
    
    def print_report(self) -> None:
        """Print a formatted report of the test results."""
        print("\n" + "=" * 80)
        print("TEST VERIFICATION REPORT")
        print("=" * 80)
        
        print("\nTest Results:")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Passed: {self.passed_tests}")
        print(f"  Failed: {self.failed_tests}")
        print(f"  Errors: {self.errors}")
        print(f"  Skipped: {self.skipped_tests}")
        
        print("\nCode Coverage:")
        print(f"  Overall Coverage: {self.overall_coverage:.2f}%")
        print(f"  Coverage Threshold: {self.coverage_threshold:.2f}%")
        
        # Print coverage by package
        if self.coverage_data:
            print("\nCoverage by Package:")
            for package, coverage in sorted(self.coverage_data.items()):
                print(f"  {package}: {coverage:.2f}%")
        
        # Print status
        tests_pass = self.failed_tests == 0 and self.errors == 0
        coverage_pass = self.overall_coverage >= self.coverage_threshold
        
        print("\nStatus:")
        if tests_pass:
            print("  All tests passed")
        else:
            print(f"  {self.failed_tests} tests failed, {self.errors} errors")
        
        if coverage_pass:
            print(f"  Coverage threshold met ({self.overall_coverage:.2f}% >= {self.coverage_threshold:.2f}%)")
        else:
            print(f"  Coverage below threshold ({self.overall_coverage:.2f}% < {self.coverage_threshold:.2f}%)")
        
        if tests_pass and coverage_pass:
            print("\nOVERALL RESULT: PASS")
        else:
            print("\nOVERALL RESULT: FAIL")
        
        print("=" * 80 + "\n")
    
    def generate_badge(self, output_dir: str = '.') -> None:
        """
        Generate a coverage badge for the README.
        
        Args:
            output_dir: Directory to save the badge
        """
        try:
            import anybadge
        except ImportError:
            print("anybadge library not installed. Install with: pip install anybadge")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate coverage badge
        badge = anybadge.Badge(
            label='coverage',
            value=f"{self.overall_coverage:.1f}%",
            thresholds={
                50: 'red',
                60: 'orange',
                80: 'yellow',
                90: 'green'
            }
        )
        
        badge.write_badge(str(output_path / 'coverage.svg'))
        
        # Generate tests badge
        if self.total_tests > 0:
            pass_rate = (self.passed_tests / self.total_tests) * 100
            badge = anybadge.Badge(
                label='tests',
                value=f"{pass_rate:.1f}%",
                thresholds={
                    50: 'red',
                    70: 'orange',
                    90: 'yellow',
                    100: 'green'
                }
            )
            
            badge.write_badge(str(output_path / 'tests.svg'))
        
        print(f"Badges saved to {output_path}")
    
    def save_report_json(self, output_file: str) -> None:
        """
        Save the test report to a JSON file.
        
        Args:
            output_file: Path to save the JSON file
        """
        report = {
            'tests': {
                'total': self.total_tests,
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'errors': self.errors,
                'skipped': self.skipped_tests,
                'pass_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'coverage': {
                'overall': self.overall_coverage,
                'threshold': self.coverage_threshold,
                'by_package': self.coverage_data
            },
            'status': {
                'tests_pass': self.failed_tests == 0 and self.errors == 0,
                'coverage_pass': self.overall_coverage >= self.coverage_threshold,
                'overall_pass': (self.failed_tests == 0 and self.errors == 0 and 
                                self.overall_coverage >= self.coverage_threshold)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {output_file}")


def main():
    """Main function to run the test verification."""
    parser = argparse.ArgumentParser(description="Verify tests and code coverage")
    parser.add_argument("--coverage-threshold", type=float, default=80.0,
                        help="Minimum acceptable code coverage percentage")
    parser.add_argument("--generate-badges", action="store_true",
                        help="Generate coverage and test badges")
    parser.add_argument("--badges-dir", default="badges",
                        help="Directory to save badges")
    parser.add_argument("--save-json", help="Save report to JSON file")
    parser.add_argument("--exit-on-fail", action="store_true",
                        help="Exit with non-zero code on failure")
    parser.add_argument("--mock-env", action="store_true",
                        help="Use mocked environment variables for testing")
    
    # Allow additional args to be passed to pytest
    parser.add_argument('pytest_args', nargs='*',
                        help="Additional arguments to pass to pytest")
    
    args = parser.parse_args()
    
    verifier = TestVerifier(
        coverage_threshold=args.coverage_threshold,
        unittest_args=args.pytest_args,
        mock_env=args.mock_env
    )
    
    success = verifier.run_tests()
    verifier.print_report()
    
    if args.generate_badges:
        verifier.generate_badge(args.badges_dir)
    
    if args.save_json:
        verifier.save_report_json(args.save_json)
    
    if args.exit_on_fail and not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
