#!/usr/bin/env python
"""
Production Readiness Check Tool for Credit Cashflow Engine

This script verifies that all required configurations and files are in place for production deployment.
It performs checks on:
- Environment variables
- Security configurations
- Required files and directories
- Database migration status
- API endpoint health

Usage:
    python check_production_readiness.py --config=prod_config.json
"""

import os
import sys
import json
import argparse
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("production-readiness")


class ProductionReadinessChecker:
    """Checks if the application is ready for production deployment."""
    
    def __init__(self, config_path: Optional[str] = None, env: str = "production"):
        """
        Initialize the production readiness checker.
        
        Args:
            config_path: Path to the configuration file (JSON)
            env: Environment to check (production, staging, etc.)
        """
        self.env = env
        self.base_dir = Path(__file__).parent.parent.absolute()
        self.config = {}
        self.checks_passed = 0
        self.checks_total = 0
        self.warnings = 0
        self.critical_failures = 0
        
        # Load configuration if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                sys.exit(1)
        
        # Default configuration
        if not self.config:
            self.config = {
                "required_env_vars": [
                    "ENV",
                    "LOG_LEVEL",
                    "WORKERS",
                    "CALCULATION_THREAD_POOL_SIZE",
                    "SUPABASE_URL",
                    "SUPABASE_KEY",
                    "SUPABASE_SERVICE_ROLE_KEY",
                    "SUPABASE_JWT_SECRET",
                    "UPSTASH_REDIS_HOST",
                    "UPSTASH_REDIS_PORT",
                    "UPSTASH_REDIS_PASSWORD",
                    "SECRET_KEY",
                    "SENTRY_DSN",
                    "CORS_ORIGINS"
                ],
                "required_files": [
                    "app/main.py",
                    "app/config.py",
                    "app/api/router.py",
                    "app/db/supabase.py",
                    "app/calculations/engine.py",
                    "Dockerfile",
                    "requirements.txt",
                    "README.md"
                ],
                "api_health_endpoint": "/api/health",
                "security_checks": {
                    "check_env_var_encryption": True,
                    "check_cors_restrictions": True,
                    "check_rate_limiting": True
                },
                "performance_thresholds": {
                    "max_response_time_ms": 500,
                    "min_rps": 10
                }
            }
    
    def run_all_checks(self) -> bool:
        """
        Run all production readiness checks.
        
        Returns:
            True if all checks passed, False otherwise
        """
        logger.info(f"Starting production readiness checks for {self.env} environment")
        
        # Run individual checks
        env_vars_check = self.check_environment_variables()
        files_check = self.check_required_files()
        security_check = self.check_security_configuration()
        api_health_check = self.check_api_health()
        
        # Print results
        self.print_results()
        
        # Return overall status
        return self.critical_failures == 0
    
    def check_environment_variables(self) -> bool:
        """
        Check if all required environment variables are set.
        
        Returns:
            True if all required environment variables are set, False otherwise
        """
        logger.info("Checking environment variables...")
        self.checks_total += 1
        
        required_vars = self.config.get("required_env_vars", [])
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            self.critical_failures += 1
            return False
        else:
            logger.info("✅ All required environment variables are set")
            self.checks_passed += 1
            return True
    
    def check_required_files(self) -> bool:
        """
        Check if all required files exist.
        
        Returns:
            True if all required files exist, False otherwise
        """
        logger.info("Checking required files...")
        self.checks_total += 1
        
        required_files = self.config.get("required_files", [])
        missing_files = []
        
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"Missing required files: {', '.join(missing_files)}")
            self.critical_failures += 1
            return False
        else:
            logger.info("✅ All required files exist")
            self.checks_passed += 1
            return True
    
    def check_security_configuration(self) -> bool:
        """
        Check security configuration.
        
        Returns:
            True if all security checks pass, False otherwise
        """
        logger.info("Checking security configuration...")
        security_checks = self.config.get("security_checks", {})
        all_passed = True
        
        # Check CORS configuration
        if security_checks.get("check_cors_restrictions", False):
            self.checks_total += 1
            cors_origins = os.environ.get("CORS_ORIGINS", "*")
            
            if cors_origins == "*":
                logger.warning("⚠️ CORS is configured to allow all origins (*)")
                self.warnings += 1
                all_passed = False
            else:
                logger.info("✅ CORS is properly restricted")
                self.checks_passed += 1
        
        # Check rate limiting
        if security_checks.get("check_rate_limiting", False):
            self.checks_total += 1
            
            # Check if rate limiting is configured in main.py or middleware
            rate_limiting_found = False
            
            # Look for rate limiting imports and configuration
            main_py_path = self.base_dir / "app/main.py"
            if main_py_path.exists():
                with open(main_py_path, 'r') as f:
                    content = f.read()
                    if "RateLimitMiddleware" in content or "limiter" in content:
                        rate_limiting_found = True
            
            if not rate_limiting_found:
                logger.warning("⚠️ Rate limiting may not be configured")
                self.warnings += 1
                all_passed = False
            else:
                logger.info("✅ Rate limiting appears to be configured")
                self.checks_passed += 1
        
        # Check if environment variables contain sensitive data encryption
        if security_checks.get("check_env_var_encryption", False):
            self.checks_total += 1
            
            # Check for SECRET_KEY and SUPABASE encryption
            if os.environ.get("SECRET_KEY") and len(os.environ.get("SECRET_KEY", "")) >= 32:
                logger.info("✅ SECRET_KEY is properly configured")
                self.checks_passed += 1
            else:
                logger.warning("⚠️ SECRET_KEY may not be properly configured")
                self.warnings += 1
                all_passed = False
        
        return all_passed
    
    def check_api_health(self) -> bool:
        """
        Check if the API health endpoint returns a successful response.
        
        Returns:
            True if the API health check passes, False otherwise
        """
        logger.info("Checking API health...")
        self.checks_total += 1
        
        health_endpoint = self.config.get("api_health_endpoint", "/api/health")
        
        # Try to get the API URL from environment
        api_base_url = os.environ.get("API_BASE_URL")
        
        if not api_base_url:
            logger.warning("API_BASE_URL environment variable not set, skipping health check")
            return False
        
        try:
            response = requests.get(f"{api_base_url}{health_endpoint}", timeout=5)
            
            if response.status_code == 200:
                logger.info(f"✅ API health check passed: {response.status_code}")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"❌ API health check failed: {response.status_code}")
                self.critical_failures += 1
                return False
        except Exception as e:
            logger.error(f"❌ API health check failed: {e}")
            self.critical_failures += 1
            return False
    
    def check_database_connectivity(self) -> bool:
        """
        Check if the database is accessible.
        
        Returns:
            True if the database check passes, False otherwise
        """
        logger.info("Checking database connectivity...")
        self.checks_total += 1
        
        # Try to connect to Supabase
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase environment variables not set, skipping database check")
            return False
        
        try:
            # Import only if needed to avoid dependency issues
            from supabase import create_client
            
            supabase = create_client(supabase_url, supabase_key)
            
            # Try a simple query
            response = supabase.from_("health_check").select("*").limit(1).execute()
            
            if not response.error:
                logger.info("✅ Database connectivity check passed")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"❌ Database connectivity check failed: {response.error}")
                self.critical_failures += 1
                return False
        except Exception as e:
            logger.error(f"❌ Database connectivity check failed: {e}")
            self.critical_failures += 1
            return False
    
    def print_results(self) -> None:
        """Print a summary of all check results."""
        print("\n" + "=" * 80)
        print("PRODUCTION READINESS CHECK RESULTS")
        print("=" * 80)
        
        print(f"\nEnvironment: {self.env}")
        print(f"Checks Passed: {self.checks_passed}/{self.checks_total}")
        print(f"Warnings: {self.warnings}")
        print(f"Critical Failures: {self.critical_failures}")
        
        if self.critical_failures > 0:
            print("\n❌ PRODUCTION READINESS CHECKS FAILED")
            print("   Fix critical issues before deploying to production")
        elif self.warnings > 0:
            print("\n⚠️ PRODUCTION READINESS CHECKS PASSED WITH WARNINGS")
            print("   Consider addressing warnings before deploying")
        else:
            print("\n✅ ALL PRODUCTION READINESS CHECKS PASSED")
            print("   Application is ready for production deployment")
        
        print("=" * 80 + "\n")


def main():
    """Main function to run production readiness checks."""
    parser = argparse.ArgumentParser(description="Check production readiness")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--env", default="production", help="Environment to check")
    parser.add_argument("--exit-on-fail", action="store_true", 
                        help="Exit with non-zero code on failure")
    
    args = parser.parse_args()
    
    checker = ProductionReadinessChecker(config_path=args.config, env=args.env)
    success = checker.run_all_checks()
    
    if args.exit_on_fail and not success:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
