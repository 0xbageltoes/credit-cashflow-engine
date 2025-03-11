"""
AbsBox Test Data Generator

This script generates sample data files for testing AbsBox integration,
creating various loan pools and deal structures with different characteristics.
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import date, datetime, timedelta

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
logger = logging.getLogger("absbox_test_data_generator")

# Asset classes and their characteristics
ASSET_CLASSES = {
    "residential_mortgage": {
        "term_range": (180, 360),
        "rate_range": (0.035, 0.065),
        "balance_range": (100000, 500000),
        "property_types": ["SingleFamily", "Condo", "Townhouse", "MultiFamily"],
        "payment_frequencies": ["Monthly"]
    },
    "auto_loan": {
        "term_range": (36, 72),
        "rate_range": (0.04, 0.09),
        "balance_range": (15000, 45000),
        "vehicle_types": ["New", "Used"],
        "payment_frequencies": ["Monthly"]
    },
    "student_loan": {
        "term_range": (120, 240),
        "rate_range": (0.03, 0.07),
        "balance_range": (10000, 100000),
        "loan_types": ["Federal", "Private"],
        "payment_frequencies": ["Monthly"]
    },
    "credit_card": {
        "term_range": (36, 60),
        "rate_range": (0.12, 0.22),
        "balance_range": (2000, 15000),
        "card_types": ["Standard", "Premium", "Rewards"],
        "payment_frequencies": ["Monthly"]
    },
    "commercial_loan": {
        "term_range": (60, 240),
        "rate_range": (0.045, 0.085),
        "balance_range": (500000, 5000000),
        "property_types": ["Office", "Retail", "Industrial", "MultiFamilyCommercial"],
        "payment_frequencies": ["Monthly", "Quarterly"]
    }
}

# Deal structures and their characteristics
DEAL_STRUCTURES = {
    "sequential": {
        "description": "Sequential payment structure",
        "min_tranches": 2,
        "max_tranches": 4
    },
    "pro_rata": {
        "description": "Pro-rata payment structure",
        "min_tranches": 2,
        "max_tranches": 3
    },
    "master_trust": {
        "description": "Master trust structure with revolving period",
        "min_tranches": 2,
        "max_tranches": 3,
        "revolving_period": (12, 24)  # months
    },
    "shifting_interest": {
        "description": "Shifting interest structure with step-down triggers",
        "min_tranches": 3,
        "max_tranches": 5
    }
}

# Scenario types
SCENARIO_TYPES = {
    "base_case": {
        "default_range": (0.01, 0.03),
        "prepayment_range": (0.05, 0.15)
    },
    "stress_case": {
        "default_range": (0.05, 0.10),
        "prepayment_range": (0.02, 0.08)
    },
    "upside_case": {
        "default_range": (0.005, 0.015),
        "prepayment_range": (0.10, 0.25)
    }
}

def generate_loan_pool(asset_class, num_loans=100, start_date=None):
    """Generate a loan pool for the given asset class
    
    Args:
        asset_class: Type of loans to generate
        num_loans: Number of loans to generate
        start_date: Start date for the loans
    
    Returns:
        Dictionary with pool data
    """
    if asset_class not in ASSET_CLASSES:
        raise ValueError(f"Unknown asset class: {asset_class}")
    
    if start_date is None:
        start_date = date.today() - timedelta(days=30)
    
    # Get characteristics for this asset class
    characteristics = ASSET_CLASSES[asset_class]
    
    # Generate loans
    loans = []
    
    for _ in range(num_loans):
        # Get random values within the ranges
        balance = round(random.uniform(*characteristics["balance_range"]), 2)
        rate = round(random.uniform(*characteristics["rate_range"]), 4)
        term = random.randint(*characteristics["term_range"])
        
        # Randomize start date slightly
        loan_start_date = start_date - timedelta(days=random.randint(0, 60))
        
        # Base loan properties
        loan = {
            "balance": balance,
            "rate": rate,
            "term": term,
            "start_date": loan_start_date.isoformat(),
            "rate_type": random.choice(["fixed", "adjustable"]) if random.random() < 0.2 else "fixed",
            "payment_frequency": random.choice(characteristics["payment_frequencies"])
        }
        
        # Add asset-specific properties
        if asset_class == "residential_mortgage":
            loan["property_type"] = random.choice(characteristics["property_types"])
            loan["ltv"] = round(random.uniform(0.6, 0.95), 2)
            loan["fico"] = random.randint(620, 800)
            
        elif asset_class == "auto_loan":
            loan["vehicle_type"] = random.choice(characteristics["vehicle_types"])
            loan["vehicle_age"] = random.randint(0, 5)
            
        elif asset_class == "student_loan":
            loan["loan_type"] = random.choice(characteristics["loan_types"])
            loan["school_type"] = random.choice(["Public", "Private"])
            
        elif asset_class == "credit_card":
            loan["card_type"] = random.choice(characteristics["card_types"])
            loan["utilization"] = round(random.uniform(0.3, 0.9), 2)
            
        elif asset_class == "commercial_loan":
            loan["property_type"] = random.choice(characteristics["property_types"])
            loan["dscr"] = round(random.uniform(1.1, 1.8), 2)
            loan["occupancy"] = round(random.uniform(0.7, 1.0), 2)
        
        loans.append(loan)
    
    # Create the pool
    pool = {
        "pool_name": f"{asset_class.replace('_', ' ').title()} Pool",
        "loans": loans
    }
    
    return pool

def generate_waterfall(deal_structure, pool_balance, start_date=None):
    """Generate a waterfall structure
    
    Args:
        deal_structure: Type of deal structure to generate
        pool_balance: Total balance of the underlying pool
        start_date: Start date for the waterfall
    
    Returns:
        Dictionary with waterfall data
    """
    if deal_structure not in DEAL_STRUCTURES:
        raise ValueError(f"Unknown deal structure: {deal_structure}")
    
    if start_date is None:
        start_date = date.today()
    
    # Get characteristics for this structure
    characteristics = DEAL_STRUCTURES[deal_structure]
    
    # Determine number of tranches
    num_tranches = random.randint(
        characteristics["min_tranches"],
        characteristics["max_tranches"]
    )
    
    # Create accounts
    accounts = [
        {
            "name": "CollectionAccount",
            "initial_balance": 0.0
        },
        {
            "name": "ReserveFund",
            "initial_balance": round(pool_balance * 0.02, 2)  # 2% reserve
        }
    ]
    
    # Create bonds (tranches)
    bonds = []
    remaining_balance = pool_balance
    
    for i in range(num_tranches):
        is_last_tranche = (i == num_tranches - 1)
        
        # Determine bond size
        if is_last_tranche:
            bond_balance = remaining_balance
        else:
            # Senior tranches get larger portions
            portion = 0.7 if i == 0 else random.uniform(0.4, 0.6)
            bond_balance = round(remaining_balance * portion, 2)
            remaining_balance -= bond_balance
        
        # Determine bond rate (higher rates for junior tranches)
        if i == 0:  # Senior
            rate = round(random.uniform(0.025, 0.04), 4)
        elif is_last_tranche:  # Most junior
            rate = round(random.uniform(0.06, 0.12), 4)
        else:  # Mezzanine
            rate = round(random.uniform(0.04, 0.06), 4)
        
        # Create the bond
        bond = {
            "name": f"Class{chr(65+i)}",  # A, B, C, etc.
            "balance": bond_balance,
            "rate": rate
        }
        
        bonds.append(bond)
    
    # Create waterfall actions
    actions = []
    
    # Add structure-specific actions
    if deal_structure == "sequential":
        # Sequential payment - pay interest to all, then principal to each class in order
        for bond in bonds:
            actions.append({
                "source": "CollectedInterest",
                "target": bond["name"],
                "amount": "Interest"
            })
        
        for bond in bonds:
            actions.append({
                "source": "CollectedPrincipal",
                "target": bond["name"],
                "amount": "OutstandingPrincipal"
            })
            
    elif deal_structure == "pro_rata":
        # Pro-rata payment - pay interest to all, then principal pro-rata
        for bond in bonds:
            actions.append({
                "source": "CollectedInterest",
                "target": bond["name"],
                "amount": "Interest"
            })
        
        for bond in bonds:
            actions.append({
                "source": "CollectedPrincipal",
                "target": bond["name"],
                "amount": "ProRataPrincipal"
            })
    
    elif deal_structure == "master_trust":
        # Master trust with revolving period
        revolving_months = random.randint(*characteristics["revolving_period"])
        
        for bond in bonds:
            actions.append({
                "source": "CollectedInterest",
                "target": bond["name"],
                "amount": "Interest"
            })
        
        # During revolving period, principal is reinvested
        actions.append({
            "source": "CollectedPrincipal",
            "target": "ReserveFund",
            "amount": "AllFunds",
            "condition": f"Period < {revolving_months}"
        })
        
        # After revolving period, pay down bonds sequentially
        for bond in bonds:
            actions.append({
                "source": "CollectedPrincipal",
                "target": bond["name"],
                "amount": "OutstandingPrincipal",
                "condition": f"Period >= {revolving_months}"
            })
            
    elif deal_structure == "shifting_interest":
        # Shifting interest structure
        for bond in bonds:
            actions.append({
                "source": "CollectedInterest",
                "target": bond["name"],
                "amount": "Interest"
            })
        
        # Initially sequential
        for bond in bonds:
            actions.append({
                "source": "CollectedPrincipal",
                "target": bond["name"],
                "amount": "OutstandingPrincipal",
                "condition": "BeforeTrigger"
            })
        
        # After trigger, switch to pro-rata for senior classes
        for i, bond in enumerate(bonds):
            if i < len(bonds) - 1:  # All but the most junior bond
                actions.append({
                    "source": "CollectedPrincipal",
                    "target": bond["name"],
                    "amount": "ProRataPrincipal",
                    "condition": "AfterTrigger"
                })
        
        # Most junior class gets remainder
        actions.append({
            "source": "CollectedPrincipal",
            "target": bonds[-1]["name"],
            "amount": "RemainingFunds",
            "condition": "AfterTrigger"
        })
    
    # Add reserve fund actions
    if len(bonds) > 0:
        # Reserve fund covers shortfalls for senior bond
        actions.append({
            "source": "ReserveFund",
            "target": bonds[0]["name"],
            "amount": "InterestShortfall"
        })
    
    # Create triggers for structures that need them
    triggers = []
    
    if deal_structure == "shifting_interest":
        # Add step-down trigger
        triggers.append({
            "name": "StepDownTrigger",
            "condition": "Period > 36 && CumulativeDefaultRate < 0.03",
            "description": "Step down trigger after 36 months if defaults are low"
        })
    
    # Create the waterfall
    waterfall = {
        "start_date": start_date.isoformat(),
        "accounts": accounts,
        "bonds": bonds,
        "actions": actions
    }
    
    if triggers:
        waterfall["triggers"] = triggers
    
    return waterfall

def generate_scenario(scenario_type, num_periods=120):
    """Generate a scenario with default and prepayment vectors
    
    Args:
        scenario_type: Type of scenario to generate
        num_periods: Number of periods in the vectors
    
    Returns:
        Dictionary with scenario data
    """
    if scenario_type not in SCENARIO_TYPES:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    # Get characteristics for this scenario
    characteristics = SCENARIO_TYPES[scenario_type]
    
    # Create default vector
    default_range = characteristics["default_range"]
    default_vector = []
    
    # Generate a curve shape based on scenario
    if scenario_type == "base_case":
        # Base case: slightly increasing then stabilizing
        for i in range(num_periods):
            period_factor = min(1.0, i / 24)  # Ramp up over 24 months
            base_default = default_range[0] + period_factor * (default_range[1] - default_range[0])
            # Add some noise
            default_rate = max(0, round(base_default + random.uniform(-0.005, 0.005), 4))
            default_vector.append(default_rate)
            
    elif scenario_type == "stress_case":
        # Stress case: rapidly increasing then gradually declining
        peak_month = random.randint(12, 36)
        for i in range(num_periods):
            if i <= peak_month:
                # Ramp up to peak
                period_factor = i / peak_month
                base_default = default_range[0] + period_factor * (default_range[1] - default_range[0])
            else:
                # Decline after peak
                period_factor = min(1.0, (i - peak_month) / 60)
                base_default = default_range[1] - period_factor * (default_range[1] - default_range[0])
            
            # Add some noise
            default_rate = max(0, round(base_default + random.uniform(-0.01, 0.01), 4))
            default_vector.append(default_rate)
            
    elif scenario_type == "upside_case":
        # Upside case: low and stable defaults
        for i in range(num_periods):
            base_default = random.uniform(*default_range)
            # Less volatility in upside case
            default_rate = max(0, round(base_default + random.uniform(-0.002, 0.002), 4))
            default_vector.append(default_rate)
    
    # Create prepayment vector
    prepay_range = characteristics["prepayment_range"]
    prepay_vector = []
    
    # Generate a curve shape based on scenario
    if scenario_type == "base_case":
        # Base case: gradual increase to plateau
        for i in range(num_periods):
            # Ramp up over 36 months then plateau
            period_factor = min(1.0, i / 36)
            base_prepay = prepay_range[0] + period_factor * (prepay_range[1] - prepay_range[0])
            # Add seasonality
            seasonality = 0.02 * math.sin(i * math.pi / 6)  # Annual cycle
            prepay_rate = max(0, round(base_prepay + seasonality + random.uniform(-0.01, 0.01), 4))
            prepay_vector.append(prepay_rate)
            
    elif scenario_type == "stress_case":
        # Stress case: lower prepayments
        for i in range(num_periods):
            base_prepay = random.uniform(*prepay_range)
            # Add some noise and seasonality
            seasonality = 0.01 * math.sin(i * math.pi / 6)
            prepay_rate = max(0, round(base_prepay + seasonality + random.uniform(-0.005, 0.005), 4))
            prepay_vector.append(prepay_rate)
            
    elif scenario_type == "upside_case":
        # Upside case: higher prepayments with refinance waves
        for i in range(num_periods):
            base_prepay = random.uniform(*prepay_range)
            
            # Add refinance waves at random intervals
            refi_wave = 0
            for wave_month in [12, 24, 48, 72]:
                if abs(i - wave_month) < 6:
                    distance = abs(i - wave_month)
                    refi_wave = 0.05 * (1 - distance / 6)
            
            prepay_rate = max(0, round(base_prepay + refi_wave + random.uniform(-0.01, 0.01), 4))
            prepay_vector.append(prepay_rate)
    
    # Create the scenario
    scenario = {
        "name": f"{scenario_type.replace('_', ' ').title()} Scenario",
        "default_curve": {
            "vector": default_vector
        },
        "prepayment_curve": {
            "vector": prepay_vector
        }
    }
    
    # Add custom assumptions based on scenario
    custom_assumptions = []
    
    if scenario_type == "base_case":
        custom_assumptions = [
            {"name": "RecoveryLag", "value": 6, "type": "integer"},
            {"name": "RecoveryRate", "value": 0.6, "type": "float"},
            {"name": "DelinquencyRate", "value": 0.025, "type": "float"}
        ]
    elif scenario_type == "stress_case":
        custom_assumptions = [
            {"name": "RecoveryLag", "value": 12, "type": "integer"},
            {"name": "RecoveryRate", "value": 0.4, "type": "float"},
            {"name": "DelinquencyRate", "value": 0.05, "type": "float"}
        ]
    elif scenario_type == "upside_case":
        custom_assumptions = [
            {"name": "RecoveryLag", "value": 4, "type": "integer"},
            {"name": "RecoveryRate", "value": 0.7, "type": "float"},
            {"name": "DelinquencyRate", "value": 0.015, "type": "float"}
        ]
    
    if custom_assumptions:
        scenario["custom_assumptions"] = custom_assumptions
    
    return scenario

def generate_test_deal(asset_class, deal_structure, scenario_type, num_loans=100, output_file=None):
    """Generate a complete test deal
    
    Args:
        asset_class: Type of loans to generate
        deal_structure: Type of deal structure to generate
        scenario_type: Type of scenario to generate
        num_loans: Number of loans to generate
        output_file: Optional file to save the deal to
    
    Returns:
        Dictionary with the complete deal
    """
    # Generate dates
    start_date = date.today() - timedelta(days=30)
    deal_date = date.today()
    
    # Generate pool
    pool = generate_loan_pool(asset_class, num_loans, start_date)
    
    # Calculate total pool balance
    total_pool_balance = sum(loan["balance"] for loan in pool["loans"])
    
    # Generate waterfall
    waterfall = generate_waterfall(deal_structure, total_pool_balance, deal_date)
    
    # Generate scenario
    scenario = generate_scenario(scenario_type)
    
    # Create the complete deal
    deal = {
        "deal_name": f"{asset_class.replace('_', ' ').title()} {deal_structure.replace('_', ' ').title()} {scenario_type.replace('_', ' ').title()}",
        "pool": pool,
        "waterfall": waterfall,
        "scenario": scenario
    }
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(deal, f, indent=2)
        logger.info(f"Deal saved to {output_file}")
    
    return deal

def generate_multiple_deals(output_dir, count=5):
    """Generate multiple test deals with various combinations
    
    Args:
        output_dir: Directory to save the deals to
        count: Number of deals to generate per combination
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate all combinations
    combinations = []
    
    for asset_class in ASSET_CLASSES.keys():
        for deal_structure in DEAL_STRUCTURES.keys():
            for scenario_type in SCENARIO_TYPES.keys():
                combinations.append((asset_class, deal_structure, scenario_type))
    
    # Generate deals for each combination
    total_deals = 0
    
    for asset_class, deal_structure, scenario_type in combinations:
        for i in range(count):
            # Vary the number of loans
            num_loans = random.randint(10, 100)
            
            # Generate a filename
            filename = f"{asset_class}_{deal_structure}_{scenario_type}_{i+1}.json"
            filepath = output_path / filename
            
            # Generate the deal
            logger.info(f"Generating deal: {filename}")
            try:
                generate_test_deal(
                    asset_class=asset_class,
                    deal_structure=deal_structure,
                    scenario_type=scenario_type,
                    num_loans=num_loans,
                    output_file=filepath
                )
                total_deals += 1
            except Exception as e:
                logger.error(f"Error generating {filename}: {e}")
    
    logger.info(f"Generated {total_deals} test deals in {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AbsBox Test Data Generator")
    
    # Operation mode
    parser.add_argument("--mode", choices=["single", "multiple"], default="single",
                       help="Generate a single deal or multiple deals")
    
    # Single deal options
    parser.add_argument("--asset-class", choices=list(ASSET_CLASSES.keys()),
                       default="residential_mortgage",
                       help="Asset class for the loans")
    parser.add_argument("--deal-structure", choices=list(DEAL_STRUCTURES.keys()),
                       default="sequential",
                       help="Structure for the deal")
    parser.add_argument("--scenario-type", choices=list(SCENARIO_TYPES.keys()),
                       default="base_case",
                       help="Type of scenario to generate")
    parser.add_argument("--num-loans", type=int, default=50,
                       help="Number of loans to generate")
    parser.add_argument("--output-file",
                       help="Output file for a single deal")
    
    # Multiple deals options
    parser.add_argument("--output-dir",
                       help="Output directory for multiple deals")
    parser.add_argument("--count", type=int, default=3,
                       help="Number of deals to generate per combination (for multiple mode)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Import math here to avoid unused import warning
    import math

    if args.mode == "single":
        # Generate a single deal
        if not args.output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = f"test_deal_{timestamp}.json"
        
        logger.info(f"Generating a single {args.asset_class} deal with {args.deal_structure} structure and {args.scenario_type} scenario")
        generate_test_deal(
            asset_class=args.asset_class,
            deal_structure=args.deal_structure,
            scenario_type=args.scenario_type,
            num_loans=args.num_loans,
            output_file=args.output_file
        )
        
    elif args.mode == "multiple":
        # Generate multiple deals
        if not args.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"test_deals_{timestamp}"
        
        logger.info(f"Generating multiple test deals in {args.output_dir}")
        generate_multiple_deals(
            output_dir=args.output_dir,
            count=args.count
        )

if __name__ == "__main__":
    main()
