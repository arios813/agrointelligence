"""Example usage and testing of the CombinationEvaluator.

Run this script to see the breeding evaluator in action:
    python3 examples/breeding_example.py
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from breeding_evaluator import CombinationEvaluator


def main():
    """Run example breeding evaluations."""
    print("=" * 70)
    print("CATTLE BREEDING GENETICS EVALUATOR - EXAMPLES")
    print("=" * 70)

    # Initialize evaluator with default champions
    evaluator = CombinationEvaluator(
        champions=["King George", "Arasunu", "Don Angel", "Nando"],
        inbreeding_penalty=15.0,
    )

    # Example combinations from the dataset
    examples = [
        ("498 x King George", "1310 x Franchesco"),
        ("678 x King George", "575 x King George"),  # Both carry King George
        ("1249 x Arasunu", "498 x King George"),      # Two different champions
        ("999 x Criminal", "388 x Nando"),            # Mixed lineage
        ("475 x Nando", "266 x Thor"),                # Nando carrier with older line
    ]

    print("\n📊 Evaluating breeding combinations:\n")

    results = []
    for parent_a, parent_b in examples:
        result = evaluator.score_combination(parent_a, parent_b)
        results.append(result)

        print(result.explanation)
        print("-" * 70)

    # Summary table
    print("\n📈 SUMMARY TABLE\n")
    print(f"{'Parent A':<25} {'Parent B':<25} {'Score':<8} {'Risk'}")
    print("-" * 70)
    for (parent_a, parent_b), result in zip(examples, results):
        risk_str = "⚠️ Yes" if result.inbreeding_risk else "✓ No"
        print(f"{parent_a:<25} {parent_b:<25} {result.score:>6.1f}   {risk_str}")

    # Top 3 combinations
    top_3 = sorted(results, key=lambda r: r.score, reverse=True)[:3]
    print(f"\n🏆 Top 3 Combinations:")
    for i, result in enumerate(top_3, 1):
        parent_a = result.parent_a_parsed["raw"]
        parent_b = result.parent_b_parsed["raw"]
        print(f"  {i}. {parent_a} × {parent_b}: {result.score:.1f}")

    # JSON export example
    print("\n📄 JSON Export (first result):")
    print(json.dumps(results[0].to_dict(), indent=2, default=str))

    # Custom champion weights example
    print("\n" + "=" * 70)
    print("ADVANCED EXAMPLE: Custom Champion Weights")
    print("=" * 70)
    evaluator_custom = CombinationEvaluator(
        champions=["King George", "Arasunu", "Don Angel"],
        champion_weights={
            "King George": 2.0,   # Premium weighting
            "Arasunu": 1.5,
            "Don Angel": 1.0,
        },
    )
    result_custom = evaluator_custom.score_combination(
        "498 x King George", "1310 x Arasunu"
    )
    print(result_custom.explanation)


if __name__ == "__main__":
    main()
