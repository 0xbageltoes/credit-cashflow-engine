#!/usr/bin/env python
"""
Production Readiness Check Script for Credit Cashflow Engine

This script checks for common issues and verifies that the application
is properly configured for production deployment.

Usage:
    python production_readiness_check.py [--env-file .env.production]
"""

import os
import sys
import json
import socket
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("production-check")

# Define paths relative to the script
BASE_DIR = Path(__file__).parent.parent.absolute()
REQUIRED_FILES = [
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    "task-definition.json",
    "requirements.txt"
]

# Define required env vars by category
SECURITY_ENV_VARS = [
    "SECRET_KEY", 
    "SUPABASE_JWT_SECRET", 
    "CORS_ORIGINS"
]

PERFORMANCE_ENV_VARS = [
    "WORKERS", 
    "CALCULATION_THREAD_POOL_SIZE", 
    "CELERY_CONCURRENCY"
]

MONITORING_ENV_VARS = [
    "PROMETHEUS_ENABLED", 
    "LOGGING_JSON_FORMAT", 
    "SENTRY_DSN"
]

CACHE_ENV_VARS = [
    "UPSTASH_REDIS_HOST", 
    "UPSTASH_REDIS_PORT", 
    "UPSTASH_REDIS_PASSWORD",
    "CACHE_TTL"
]

# Score weights for final rating
WEIGHTS = {
    "security": 0.3,
    "performance": 0.2,
    "monitoring": 0.2,
    "dependencies": 0.1,
    "code_quality": 0.1,
    "docker": 0.1
}

class ProductionReadinessChecker:
    """Verifies production readiness of the application."""
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file
        self.env_vars = {}
        self.issues = []
        self.warnings = []
        self.score = {
            "security": 0,
            "performance": 0,
            "monitoring": 0,
            "dependencies": 0,
            "code_quality": 0,
            "docker": 0
        }
        
        # Load environment variables if provided
        if env_file:
            self._load_env_file()
    
    def _load_env_file(self) -> None:
        """Load environment variables from the .env file."""
        logger.info(f"Loading environment variables from {self.env_file}")
        
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    key, value = line.split('=', 1)
                    self.env_vars[key.strip()] = value.strip()
            
            logger.info(f"Loaded {len(self.env_vars)} environment variables")
        
        except Exception as e:
            logger.error(f"Failed to load environment file: {e}")
            self.issues.append(f"Failed to load environment file: {e}")
    
    def check_required_files(self) -> None:
        """Check that all required files exist."""
        logger.info("Checking for required files...")
        
        for file_path in REQUIRED_FILES:
            full_path = BASE_DIR / file_path
            if not full_path.exists():
                self.issues.append(f"Missing required file: {file_path}")
                logger.warning(f"Missing required file: {file_path}")
            else:
                logger.info(f"Found required file: {file_path}")
    
    def check_environment_variables(self) -> None:
        """Check that all required environment variables are set."""
        logger.info("Checking environment variables...")
        
        # Check security-related env vars
        security_score = self._check_env_var_category(SECURITY_ENV_VARS, "security")
        self.score["security"] = security_score
        
        # Check performance-related env vars
        performance_score = self._check_env_var_category(PERFORMANCE_ENV_VARS, "performance")
        self.score["performance"] = performance_score
        
        # Check monitoring-related env vars
        monitoring_score = self._check_env_var_category(MONITORING_ENV_VARS, "monitoring")
        self.score["monitoring"] = monitoring_score
        
        # Check cache-related env vars
        cache_score = self._check_env_var_category(CACHE_ENV_VARS, "cache")
        # Cache contributes to performance score
        self.score["performance"] = (self.score["performance"] + cache_score) / 2
        
        # Check for DEBUG or development mode
        if self.env_vars.get("ENV", "").lower() != "production":
            self.issues.append("ENV is not set to 'production'")
            self.score["security"] *= 0.7
        
        if self.env_vars.get("DEBUG", "").lower() in ["true", "1", "yes"]:
            self.issues.append("DEBUG mode is enabled in a production environment")
            self.score["security"] *= 0.5
    
    def _check_env_var_category(self, vars_list: List[str], category: str) -> float:
        """Check a specific category of environment variables."""
        found = 0
        for var in vars_list:
            if var in self.env_vars and self.env_vars[var]:
                found += 1
                logger.info(f"Found {category} variable: {var}")
            else:
                self.issues.append(f"Missing {category} variable: {var}")
                logger.warning(f"Missing {category} variable: {var}")
        
        return found / len(vars_list) if vars_list else 1.0
    
    def check_dependencies(self) -> None:
        """Check dependencies for security vulnerabilities."""
        logger.info("Checking dependencies for vulnerabilities...")
        
        # Check if safety is installed
        try:
            subprocess.run(["safety", "--version"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           check=True)
            
            req_file = BASE_DIR / "requirements.txt"
            if req_file.exists():
                # Run safety check on requirements
                process = subprocess.run(
                    ["safety", "check", "-r", str(req_file), "--json"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if process.returncode == 0:
                    logger.info("No security vulnerabilities found in dependencies")
                    self.score["dependencies"] = 1.0
                else:
                    # Parse JSON output from safety
                    try:
                        result = json.loads(process.stdout.decode('utf-8'))
                        if result.get("vulnerabilities"):
                            vuln_count = len(result["vulnerabilities"])
                            self.issues.append(f"Found {vuln_count} security vulnerabilities in dependencies")
                            logger.warning(f"Found {vuln_count} security vulnerabilities in dependencies")
                            
                            # Adjust score based on number of vulnerabilities
                            self.score["dependencies"] = max(0, 1 - (vuln_count * 0.1))
                    except json.JSONDecodeError:
                        self.warnings.append("Could not parse safety check results")
                        self.score["dependencies"] = 0.5
            else:
                self.issues.append("requirements.txt file not found")
                self.score["dependencies"] = 0
        
        except (subprocess.SubprocessError, FileNotFoundError):
            self.warnings.append("Safety package not installed. Cannot check for vulnerabilities.")
            self.score["dependencies"] = 0.5
            logger.warning("Safety package not installed. Install with: pip install safety")
    
    def check_code_quality(self) -> None:
        """Check code quality using pylint or similar tool."""
        logger.info("Checking code quality...")
        
        # Check if pylint is installed
        try:
            subprocess.run(["pylint", "--version"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           check=True)
            
            # Run pylint on the app directory
            app_dir = BASE_DIR / "app"
            if app_dir.exists():
                process = subprocess.run(
                    ["pylint", str(app_dir), "--exit-zero", "--output-format=json"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Parse JSON output from pylint
                try:
                    results = json.loads(process.stdout.decode('utf-8'))
                    error_count = len([r for r in results if r.get("type") in ("error", "fatal")])
                    warning_count = len([r for r in results if r.get("type") == "warning"])
                    
                    if error_count > 0:
                        self.issues.append(f"Found {error_count} code quality errors")
                        logger.warning(f"Found {error_count} code quality errors")
                    
                    if warning_count > 0:
                        self.warnings.append(f"Found {warning_count} code quality warnings")
                        logger.info(f"Found {warning_count} code quality warnings")
                    
                    # Calculate score based on errors and warnings
                    if error_count + warning_count > 0:
                        error_weight = 2  # Errors count more than warnings
                        total_weight = error_count * error_weight + warning_count
                        self.score["code_quality"] = max(0, 1 - (total_weight / 100))
                    else:
                        self.score["code_quality"] = 1.0
                        logger.info("No code quality issues found")
                
                except json.JSONDecodeError:
                    self.warnings.append("Could not parse pylint results")
                    self.score["code_quality"] = 0.5
            else:
                self.warnings.append("app directory not found")
                self.score["code_quality"] = 0.5
        
        except (subprocess.SubprocessError, FileNotFoundError):
            self.warnings.append("Pylint not installed. Cannot check code quality.")
            self.score["code_quality"] = 0.5
            logger.warning("Pylint not installed. Install with: pip install pylint")
    
    def check_docker_configuration(self) -> None:
        """Check Docker configuration for best practices."""
        logger.info("Checking Docker configuration...")
        
        dockerfile_path = BASE_DIR / "Dockerfile"
        if not dockerfile_path.exists():
            self.issues.append("Dockerfile not found")
            self.score["docker"] = 0
            return
        
        # Check Dockerfile for best practices
        try:
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            # Check for non-root user
            if "USER" in dockerfile_content:
                logger.info("Dockerfile uses non-default USER")
            else:
                self.warnings.append("Dockerfile does not set a non-root USER")
                self.score["docker"] = max(0.7, self.score["docker"])
            
            # Check for HEALTHCHECK
            if "HEALTHCHECK" in dockerfile_content:
                logger.info("Dockerfile includes HEALTHCHECK")
            else:
                self.warnings.append("Dockerfile does not include HEALTHCHECK directive")
                self.score["docker"] = max(0.8, self.score["docker"])
            
            # Check for proper base image
            if "python:slim" in dockerfile_content or "python:alpine" in dockerfile_content:
                logger.info("Dockerfile uses slim/alpine base image")
            else:
                self.warnings.append("Consider using a smaller base image like python:slim or python:alpine")
            
            # Set docker score based on checks
            self.score["docker"] = 1.0
        
        except Exception as e:
            self.warnings.append(f"Error checking Dockerfile: {e}")
            self.score["docker"] = 0.5
    
    def check_network_connectivity(self) -> None:
        """Check network connectivity to required services."""
        logger.info("Checking network connectivity...")
        
        # Check Redis connection
        redis_host = self.env_vars.get("UPSTASH_REDIS_HOST")
        redis_port = int(self.env_vars.get("UPSTASH_REDIS_PORT", "6379"))
        
        if redis_host:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((redis_host, redis_port))
                if result == 0:
                    logger.info(f"Successfully connected to Redis at {redis_host}:{redis_port}")
                else:
                    self.issues.append(f"Cannot connect to Redis at {redis_host}:{redis_port}")
                    logger.warning(f"Cannot connect to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                self.warnings.append(f"Error checking Redis connection: {e}")
                logger.warning(f"Error checking Redis connection: {e}")
            finally:
                sock.close()
    
    def calculate_overall_score(self) -> float:
        """Calculate the overall production readiness score."""
        overall_score = 0
        for category, weight in WEIGHTS.items():
            category_score = self.score.get(category, 0)
            overall_score += category_score * weight
        
        return overall_score * 100  # Convert to percentage
    
    def run_all_checks(self) -> float:
        """Run all checks and return the overall score."""
        logger.info("Starting production readiness checks...")
        
        self.check_required_files()
        if self.env_file:
            self.check_environment_variables()
            self.check_network_connectivity()
        self.check_dependencies()
        self.check_code_quality()
        self.check_docker_configuration()
        
        overall_score = self.calculate_overall_score()
        
        return overall_score
    
    def print_report(self) -> None:
        """Print a report of the production readiness checks."""
        overall_score = self.calculate_overall_score()
        
        print("\n" + "=" * 80)
        print(f"PRODUCTION READINESS REPORT")
        print("=" * 80)
        
        print(f"\nOverall Score: {overall_score:.1f}%")
        
        if overall_score >= 90:
            rating = "EXCELLENT"
        elif overall_score >= 75:
            rating = "GOOD"
        elif overall_score >= 60:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        print(f"Rating: {rating}")
        
        # Print category scores
        print("\nCategory Scores:")
        for category, weight in WEIGHTS.items():
            score = self.score.get(category, 0) * 100
            print(f"  - {category.title()}: {score:.1f}% (weight: {weight*100:.0f}%)")
        
        # Print issues
        if self.issues:
            print("\nIssues that must be addressed:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        # Print warnings
        if self.warnings:
            print("\nWarnings to consider:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Recommendations
        print("\nRecommendations:")
        if overall_score < 60:
            print("  - Address all issues before deploying to production")
            print("  - Set up proper monitoring and logging")
            print("  - Ensure all security-related environment variables are configured")
        elif overall_score < 75:
            print("  - Address critical issues related to security and performance")
            print("  - Consider implementing additional monitoring")
        elif overall_score < 90:
            print("  - Address warnings to improve production readiness")
            print("  - Review security best practices")
        else:
            print("  - Your application is well-prepared for production deployment")
            print("  - Continue monitoring and updating dependencies regularly")
        
        print("\nNOTE: This check does not guarantee that your application is ready for production.")
        print("      It is meant to be a guideline for common production readiness concerns.")
        print("=" * 80 + "\n")


def main():
    """Main function to run production readiness checks."""
    parser = argparse.ArgumentParser(description="Check production readiness of the application")
    parser.add_argument("--env-file", default=None, help="Path to the environment file")
    args = parser.parse_args()
    
    env_file = args.env_file
    if env_file:
        env_path = Path(env_file)
        if not env_path.is_absolute():
            env_path = BASE_DIR / env_file
        env_file = str(env_path)
    
    checker = ProductionReadinessChecker(env_file)
    checker.run_all_checks()
    checker.print_report()


if __name__ == "__main__":
    main()
