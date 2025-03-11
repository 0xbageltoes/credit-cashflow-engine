"""
AbsBox Load Testing Script

This script performs load testing on the AbsBox integration by simulating
multiple concurrent requests with varying complexity.
"""
import sys
import os
import time
import json
import random
import argparse
import concurrent.futures
import logging
import statistics
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

# Import after setting path
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("absbox_load_test")

# Try to import AbsBox service
try:
    from app.services.absbox_service import AbsBoxService
    from app.models.structured_products import (
        StructuredDealRequest,
        LoanPoolConfig,
        LoanConfig,
        WaterfallConfig,
        AccountConfig,
        BondConfig,
        WaterfallAction,
        ScenarioConfig,
        DefaultCurveConfig,
        PrepaymentCurveConfig
    )
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you have installed the required dependencies with:")
    logger.error("pip install -r requirements.txt")
    sys.exit(1)

def create_test_deal(complexity="medium", random_seed=None):
    """Create a test deal with varying complexity
    
    Args:
        complexity: "low", "medium", or "high"
        random_seed: Optional seed for reproducibility
    
    Returns:
        StructuredDealRequest object
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    today = date.today()
    start_date = today - timedelta(days=30)
    
    # Configure parameters based on complexity
    if complexity == "low":
        num_loans = 5
        num_bonds = 2
        num_periods = 5
    elif complexity == "high":
        num_loans = 50
        num_bonds = 5
        num_periods = 20
    else:  # medium
        num_loans = 20
        num_bonds = 3
        num_periods = 10
    
    # Create loans
    loans = []
    for i in range(num_loans):
        balance = random.uniform(100000, 500000)
        rate = random.uniform(0.03, 0.06)
        term = random.choice([120, 180, 240, 360])
        
        loan = LoanConfig(
            balance=round(balance, 2),
            rate=round(rate, 4),
            term=term,
            start_date=start_date - timedelta(days=random.randint(0, 60)),
            rate_type="fixed",
            payment_frequency="Monthly"
        )
        loans.append(loan)
    
    # Create pool
    pool_config = LoanPoolConfig(
        pool_name=f"{complexity.capitalize()} Complexity Pool",
        loans=loans
    )
    
    # Calculate total pool balance
    total_pool_balance = sum(loan.balance for loan in loans)
    
    # Create bonds
    bonds = []
    remaining_balance = total_pool_balance
    
    for i in range(num_bonds):
        is_last_bond = (i == num_bonds - 1)
        
        if is_last_bond:
            bond_balance = remaining_balance
        else:
            # Allocate a portion of the remaining balance
            portion = random.uniform(0.4, 0.7) if i == 0 else random.uniform(0.3, 0.5)
            bond_balance = remaining_balance * portion
            remaining_balance -= bond_balance
        
        # Higher rate for junior bonds
        if i == 0:  # Senior
            rate = random.uniform(0.03, 0.04)
        elif is_last_bond:  # Most junior
            rate = random.uniform(0.06, 0.08)
        else:  # Mezzanine
            rate = random.uniform(0.04, 0.06)
        
        bond = BondConfig(
            name=f"Class{chr(65+i)}",  # A, B, C, etc.
            balance=round(bond_balance, 2),
            rate=round(rate, 4)
        )
        bonds.append(bond)
    
    # Create waterfall actions
    actions = []
    
    # Interest payments - always sequential
    for bond in bonds:
        actions.append(
            WaterfallAction(
                source="CollectedInterest",
                target=bond.name,
                amount="Interest"
            )
        )
    
    # Principal payments - sequential
    for bond in bonds:
        actions.append(
            WaterfallAction(
                source="CollectedPrincipal",
                target=bond.name,
                amount="OutstandingPrincipal"
            )
        )
    
    # Create waterfall
    waterfall_config = WaterfallConfig(
        start_date=today,
        accounts=[
            AccountConfig(
                name="ReserveFund",
                initial_balance=total_pool_balance * 0.02  # 2% reserve
            )
        ],
        bonds=bonds,
        actions=actions
    )
    
    # Create default and prepayment curves
    default_vector = [round(random.uniform(0.005, 0.03), 4) for _ in range(num_periods)]
    prepay_vector = [round(random.uniform(0.02, 0.15), 4) for _ in range(num_periods)]
    
    # Create scenario
    scenario_config = ScenarioConfig(
        name=f"{complexity.capitalize()} Scenario",
        default_curve=DefaultCurveConfig(
            vector=default_vector
        ),
        prepayment_curve=PrepaymentCurveConfig(
            vector=prepay_vector
        )
    )
    
    # Create the deal request
    deal_request = StructuredDealRequest(
        deal_name=f"{complexity.capitalize()} Complexity Test Deal",
        pool=pool_config,
        waterfall=waterfall_config,
        scenario=scenario_config
    )
    
    return deal_request

def test_worker(service, complexity, worker_id, use_cache=True):
    """Worker function to run a test deal
    
    Args:
        service: AbsBoxService instance
        complexity: "low", "medium", or "high"
        worker_id: Worker ID for logging
        use_cache: Whether to use cache
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"Worker {worker_id}: Starting {complexity} complexity test")
    
    # Create a deal with a seed based on worker_id for reproducibility
    deal_request = create_test_deal(complexity=complexity, random_seed=worker_id)
    
    # Disable cache if requested
    if not use_cache:
        # Add a random suffix to the deal name to prevent cache hits
        deal_request.deal_name = f"{deal_request.deal_name}_{random.randint(1, 1000000)}"
    
    start_time = time.time()
    success = False
    error_msg = None
    result = None
    
    try:
        result = service.analyze_deal(deal_request)
        if result.status == "success":
            success = True
        else:
            error_msg = result.error
    except Exception as e:
        error_msg = str(e)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Process results
    test_result = {
        "worker_id": worker_id,
        "complexity": complexity,
        "deal_name": deal_request.deal_name,
        "num_loans": len(deal_request.pool.loans),
        "num_bonds": len(deal_request.waterfall.bonds),
        "success": success,
        "execution_time": execution_time,
        "service_reported_time": result.execution_time if result and hasattr(result, "execution_time") else None,
        "error": error_msg
    }
    
    if success:
        test_result["cashflow_periods"] = len(result.bond_cashflows) if hasattr(result, "bond_cashflows") else 0
    
    logger.info(
        f"Worker {worker_id}: {complexity} test completed in {execution_time:.2f}s, "
        f"success: {success}"
    )
    
    return test_result

def run_load_test(num_workers=4, complexity="medium", use_cache=True, output_file=None):
    """Run a load test with multiple workers
    
    Args:
        num_workers: Number of concurrent workers
        complexity: "low", "medium", "high", or "mixed"
        use_cache: Whether to use cache
        output_file: Optional JSON file to save results
    
    Returns:
        List of test results
    """
    logger.info(f"Starting load test with {num_workers} workers, "
               f"complexity: {complexity}, cache: {'enabled' if use_cache else 'disabled'}")
    
    # Initialize service
    service = AbsBoxService()
    
    # Check health before starting
    try:
        health = service.health_check()
        logger.info(f"Service health: {health['status']}")
        logger.info(f"Engine type: {health['engine']['type']}")
        logger.info(f"Cache: {health['cache']['status']}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return []
    
    # Prepare worker parameters
    worker_params = []
    
    if complexity == "mixed":
        # Distribute workers across different complexity levels
        complexities = ["low"] * (num_workers // 3) + ["medium"] * (num_workers // 3) + ["high"] * (num_workers // 3)
        # Fill any remaining workers with medium complexity
        while len(complexities) < num_workers:
            complexities.append("medium")
    else:
        complexities = [complexity] * num_workers
    
    for i in range(num_workers):
        worker_params.append((service, complexities[i], i, use_cache))
    
    # Run tests in parallel
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(test_worker, *params) for params in worker_params]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Worker failed: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Compile statistics
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    logger.info(f"Load test completed in {total_time:.2f} seconds")
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful tests: {len(successful_tests)}")
    logger.info(f"Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        execution_times = [r["execution_time"] for r in successful_tests]
        logger.info(f"Average execution time: {statistics.mean(execution_times):.2f}s")
        logger.info(f"Median execution time: {statistics.median(execution_times):.2f}s")
        logger.info(f"Min execution time: {min(execution_times):.2f}s")
        logger.info(f"Max execution time: {max(execution_times):.2f}s")
    
    # Save results if requested
    if output_file:
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "num_workers": num_workers,
                "complexity": complexity,
                "cache_enabled": use_cache
            },
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "total_time": total_time
            },
            "statistics": {
                "execution_time": {
                    "mean": statistics.mean(execution_times) if successful_tests else None,
                    "median": statistics.median(execution_times) if successful_tests else None,
                    "min": min(execution_times) if successful_tests else None,
                    "max": max(execution_times) if successful_tests else None
                }
            },
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return results

def analyze_results(results, output_dir=None):
    """Analyze test results and generate plots
    
    Args:
        results: List of test results
        output_dir: Optional directory to save plots
    """
    if not results:
        logger.error("No results to analyze")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # 1. Execution time by complexity
    plt.figure(figsize=(10, 6))
    df_success = df[df["success"]]
    
    if "complexity" in df_success.columns and not df_success.empty:
        df_success.boxplot(column="execution_time", by="complexity")
        plt.title("Execution Time by Complexity")
        plt.suptitle("")  # Remove default title
        plt.xlabel("Complexity")
        plt.ylabel("Execution Time (s)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_dir:
            plt.savefig(output_path / "execution_time_by_complexity.png")
        plt.close()
    
    # 2. Success rate by complexity
    plt.figure(figsize=(10, 6))
    
    if "complexity" in df.columns:
        success_rate = df.groupby("complexity")["success"].mean() * 100
        success_rate.plot(kind="bar")
        plt.title("Success Rate by Complexity")
        plt.xlabel("Complexity")
        plt.ylabel("Success Rate (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 100)
        
        # Add value labels
        for i, value in enumerate(success_rate):
            plt.text(i, value + 2, f"{value:.1f}%", ha="center")
        
        if output_dir:
            plt.savefig(output_path / "success_rate_by_complexity.png")
        plt.close()
    
    # 3. Execution time histogram
    plt.figure(figsize=(10, 6))
    
    if not df_success.empty:
        plt.hist(df_success["execution_time"], bins=20, alpha=0.7)
        plt.title("Distribution of Execution Times")
        plt.xlabel("Execution Time (s)")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add summary statistics
        mean_time = df_success["execution_time"].mean()
        median_time = df_success["execution_time"].median()
        min_time = df_success["execution_time"].min()
        max_time = df_success["execution_time"].max()
        
        stats_text = (
            f"Mean: {mean_time:.2f}s\n"
            f"Median: {median_time:.2f}s\n"
            f"Min: {min_time:.2f}s\n"
            f"Max: {max_time:.2f}s"
        )
        
        plt.annotate(stats_text, xy=(0.7, 0.7), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        if output_dir:
            plt.savefig(output_path / "execution_time_histogram.png")
        plt.close()
    
    # 4. Error analysis
    failures = df[~df["success"]]
    
    if not failures.empty:
        logger.info(f"Error analysis ({len(failures)} failures):")
        
        # Group by error message
        error_counts = failures["error"].value_counts()
        logger.info("Top errors:")
        for error, count in error_counts.items():
            logger.info(f"  {count} occurrences: {error}")
        
        # Error rate by complexity
        if "complexity" in failures.columns:
            error_by_complexity = failures.groupby("complexity").size()
            total_by_complexity = df.groupby("complexity").size()
            error_rate = (error_by_complexity / total_by_complexity * 100).fillna(0)
            
            logger.info("Error rate by complexity:")
            for complexity, rate in error_rate.items():
                logger.info(f"  {complexity}: {rate:.1f}%")
    
    # Summary report
    if output_dir:
        with open(output_path / "summary_report.txt", "w") as f:
            f.write("AbsBox Load Test Summary Report\n")
            f.write("=============================\n\n")
            
            f.write(f"Total tests: {len(df)}\n")
            f.write(f"Successful tests: {len(df_success)}\n")
            f.write(f"Failed tests: {len(failures)}\n")
            f.write(f"Overall success rate: {(len(df_success) / len(df) * 100):.1f}%\n\n")
            
            if "complexity" in df.columns:
                f.write("Success rate by complexity:\n")
                for complexity, rate in success_rate.items():
                    f.write(f"  {complexity}: {rate:.1f}%\n")
                f.write("\n")
            
            if not df_success.empty:
                f.write("Execution time statistics:\n")
                f.write(f"  Mean: {mean_time:.2f}s\n")
                f.write(f"  Median: {median_time:.2f}s\n")
                f.write(f"  Min: {min_time:.2f}s\n")
                f.write(f"  Max: {max_time:.2f}s\n\n")
            
            if not failures.empty:
                f.write(f"Error analysis ({len(failures)} failures):\n")
                f.write("Top errors:\n")
                for error, count in error_counts.items():
                    f.write(f"  {count} occurrences: {error}\n")
                f.write("\n")
        
        logger.info(f"Summary report saved to {output_path / 'summary_report.txt'}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AbsBox Load Testing Tool")
    
    # Test configuration
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of concurrent workers (default: 4)")
    parser.add_argument("--complexity", "-c", choices=["low", "medium", "high", "mixed"],
                       default="medium", help="Test complexity (default: medium)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--output", "-o", 
                       help="Output directory for results and plots")
    parser.add_argument("--iterations", "-i", type=int, default=1,
                       help="Number of test iterations to run (default: 1)")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Results will be saved to {output_dir}")
    
    # Run multiple iterations if requested
    all_results = []
    
    for iteration in range(args.iterations):
        if args.iterations > 1:
            logger.info(f"Starting iteration {iteration + 1}/{args.iterations}")
        
        # Determine output file for this iteration
        output_file = None
        if output_dir:
            if args.iterations > 1:
                output_file = output_dir / f"results_iteration_{iteration + 1}.json"
            else:
                output_file = output_dir / "results.json"
        
        # Run the load test
        results = run_load_test(
            num_workers=args.workers,
            complexity=args.complexity,
            use_cache=not args.no_cache,
            output_file=output_file
        )
        
        all_results.extend(results)
        
        if args.iterations > 1:
            logger.info(f"Completed iteration {iteration + 1}/{args.iterations}")
            
            # Small delay between iterations
            if iteration < args.iterations - 1:
                time.sleep(2)
    
    # Analyze combined results
    if all_results:
        analyze_results(all_results, output_dir)

if __name__ == "__main__":
    main()
