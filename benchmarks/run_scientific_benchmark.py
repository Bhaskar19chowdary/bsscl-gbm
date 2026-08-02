import argparse
import sys
from suite_data_scaling import run_data_scaling_suite
from suite_data_quality import run_data_quality_suite
from suite_learning_capability import run_learning_capability_suite
from suite_performance import run_performance_suite
from suite_enterprise import run_enterprise_suite

def main():
    parser = argparse.ArgumentParser(description="Run the 45-point Scientific Benchmark Suite")
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (lower dataset sizes)')
    args = parser.parse_args()
    
    print("\n" + "#" * 70)
    print("  BSSCL-GBM SCIENTIFIC & INDUSTRIAL BENCHMARK SUITE")
    print("  Executing comprehensive 45-category evaluation...")
    if args.fast:
        print("  MODE: FAST (--fast flag enabled)")
    else:
        print("  MODE: FULL (Warning: May take several hours)")
    print("#" * 70 + "\n")

    try:
        run_data_scaling_suite(fast_mode=args.fast)
        run_data_quality_suite(fast_mode=args.fast)
        run_learning_capability_suite(fast_mode=args.fast)
        run_performance_suite(fast_mode=args.fast)
        run_enterprise_suite(fast_mode=args.fast)
        
        print("\n" + "#" * 70)
        print("✅ ALL BENCHMARK SUITES COMPLETED SUCCESSFULLY!")
        print("#" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Benchmark Suite Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
