"""Breeding Genetics Evaluator - Summary

This module provides tools to score and evaluate proposed cattle breeding combinations
based on:
1. Champion/elite bloodline presence
2. Genetic diversity and inbreeding risk
3. Historical lineage data

Files:
- src/breeding_evaluator.py — Core CombinationEvaluator class
- examples/breeding_example.py — Example usage and testing
- src/web_app.py — Integrated Streamlit UI (new section)

Quick Start
===========

1. Load evaluator with default champions:
   evaluator = CombinationEvaluator()
   result = evaluator.score_combination("498 x King George", "1310 x Arasunu")
   print(result)

2. Or with custom champion weights:
   evaluator = CombinationEvaluator(
       champions=["King George", "Arasunu", "Don Angel"],
       champion_weights={"King George": 2.0, "Arasunu": 1.5}
   )

3. Evaluate multiple combinations at once:
   combinations = [
       ("498 x King George", "1310 x Franchesco"),
       ("678 x King George", "575 x King George"),
   ]
   results = evaluator.bulk_evaluate(combinations)

API Reference
=============

CombinationEvaluator
--------------------
Methods:
  - __init__(champions, inbreeding_penalty, champion_weights)
  - parse_lineage(lineage_str) -> Dict  # Extract sire/dam from string
  - detect_champions(lineage_dict) -> Dict  # Find champion matches
  - score_combination(parent_a, parent_b) -> ScoringResult
  - bulk_evaluate(combinations) -> List[ScoringResult]

ScoringResult (dataclass)
---------
Fields:
  - score (float): 0-100 normalized score
  - explanation (str): Human-readable summary
  - champion_matches (Dict): {champion: count}
  - inbreeding_risk (bool): Duplicate sire detected
  - parent_a_parsed (Dict): Parsed sire/dam for parent A
  - parent_b_parsed (Dict): Parsed sire/dam for parent B

Scoring Logic
=============

Score calculation (raw, before normalization):
  1. Base points: +10 per champion match (weighted)
  2. Bonus: +5 per additional occurrence of same champion (weighted)
  3. Homozygous bonus: +5 if same champion in both parents
  4. Inbreeding penalty: -15 per detected duplicate sire
  5. Final: Clamp to 0-100

Score interpretation:
  - 80-100: Excellent genetic combination
  - 60-79: Good genetic foundation
  - 40-59: Moderate genetic profile
  - 0-39: Limited champion genetics

Extensibility
==============

1. Custom champion list:
   evaluator = CombinationEvaluator(
       champions=["Your Line", "Another Elite", "Third Champion"]
   )

2. Weighted importance:
   evaluator = CombinationEvaluator(
       champion_weights={
           "Your Line": 3.0,  # 3x weight
           "Another Elite": 1.5,
       }
   )

3. Advanced parsing:
   Override parse_lineage() to handle custom formats (e.g., "ID|Name").

4. Custom scoring:
   Subclass CombinationEvaluator and override calculate_score().

Streamlit UI Integration
=========================

The web_app.py includes an interactive breeding genetics section where users can:
- Select lineages from the dataset dropdown
- Enter custom lineages manually
- View color-coded scores (green=good, orange=moderate, red=poor)
- See detected champions and inbreeding risks
- Inspect parsed lineage details

To launch:
  streamlit run src/web_app.py

Then navigate to "Breeding Genetics Evaluator" section.

Testing
=======

Run the example to see the evaluator in action:
  python3 examples/breeding_example.py

This generates:
- Evaluation of 5 example combinations
- Summary table (score + risk)
- Top 3 ranked combinations
- JSON export example  
- Advanced example with custom weights

Notes
=====

- Lineage parsing is simple pattern-matching; complex genealogies may need
  a more sophisticated pedigree database.
- Inbreeding detection is basic (duplicate sires only); for real genetics,
  consider coefficient of inbreeding (COI) or F-statistics.
- Scores are normalized 0-100 for user-friendliness; internal scoring can exceed this.
- Champion list and weights should be updated by breed experts for production use.

Author Note
===========

This module is designed for extensibility and production integration. 
Future improvements could include:
- Pedigree tree visualization
- COI calculation
- Multi-generation tracking
- Integration with genetic databases (e.g., EPD, genomic data)
- Real-time model scoring using ML predictions from the main pipeline
"""
