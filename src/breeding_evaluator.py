"""Breeding combination evaluator for cattle genetics.

Provides functionality to score proposed breeding combinations based on:
- Presence of champion bloodlines
- Frequency of champion genetics
- Basic inbreeding detection (duplicate sire penalty)

Usage:
    evaluator = CombinationEvaluator(champions=['King George', 'Arasunu', 'Don Angel'])
    result = evaluator.score_combination('498 x King George', '1310 x Arasunu')
    print(result)
"""
from typing import List, Dict, Tuple, Optional
import re
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of a breeding combination evaluation."""
    score: float  # 0-100
    explanation: str
    champion_matches: Dict[str, int]  # {champion_name: count}
    inbreeding_risk: bool
    parent_a_parsed: Dict[str, any]
    parent_b_parsed: Dict[str, any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CombinationEvaluator:
    """Evaluates breeding combinations based on historical champion genetics.
    
    Attributes:
        champions (List[str]): Names of champion/elite sires/lines.
        max_score (float): Maximum possible base score (before normalization).
        inbreeding_penalty (float): Penalty for detected inbreeding loops.
        champion_weights (Dict[str, float]): Optional custom weights per champion.
    """

    def __init__(
        self,
        champions: Optional[List[str]] = None,
        inbreeding_penalty: float = 15.0,
        champion_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize evaluator.
        
        Parameters
        ----------
        champions : List[str], optional
            Names of elite/champion sires. Defaults to known breeds.
        inbreeding_penalty : float
            Score penalty for each detected duplicate sire (0-100 scale).
        champion_weights : Dict[str, float], optional
            Custom multiplier for each champion (e.g., {'King George': 2.0}).
        """
        self.champions = champions or [
            "King George",
            "Arasunu",
            "Don Angel",
            "Nando",
            "Franchesco",
            "Spartacus",
        ]
        self.inbreeding_penalty = inbreeding_penalty
        self.champion_weights = champion_weights or {c: 1.0 for c in self.champions}
        logger.debug(f"Initialized CombinationEvaluator with champions: {self.champions}")

    def parse_lineage(self, lineage_str: str) -> Dict[str, Optional[str]]:
        """Parse a lineage string to extract sire and dam references.
        
        Expected format: "ID x Name" or just "Name", case-insensitive.
        
        Parameters
        ----------
        lineage_str : str
            Lineage string, e.g., "498 x King George"
        
        Returns
        -------
        Dict[str, Optional[str]]
            {
                'raw': original string,
                'sire': extracted sire name,
                'dam': extracted dam name,
            }
        """
        if not lineage_str or pd.isna(lineage_str):
            return {"raw": "", "sire": None, "dam": None}

        lineage_str = str(lineage_str).strip()
        parts = [p.strip() for p in lineage_str.split("x")]

        result = {"raw": lineage_str, "sire": None, "dam": None}

        if len(parts) == 2:
            result["sire"] = parts[0] if parts[0] else None
            result["dam"] = parts[1] if parts[1] else None
        elif len(parts) == 1:
            result["sire"] = parts[0] if parts[0] else None

        return result

    def detect_champions(self, lineage_dict: Dict) -> Dict[str, int]:
        """Detect champion presence in a parsed lineage.
        
        Parameters
        ----------
        lineage_dict : Dict
            Output of parse_lineage().
        
        Returns
        -------
        Dict[str, int]
            {champion_name: count_in_lineage}
        """
        matches = {c: 0 for c in self.champions}
        raw = lineage_dict.get("raw", "").lower()

        for champion in self.champions:
            # Count occurrences (case-insensitive)
            count = raw.lower().count(champion.lower())
            if count > 0:
                matches[champion] = count

        return {k: v for k, v in matches.items() if v > 0}

    def calculate_score(
        self,
        parent_a_parsed: Dict,
        parent_b_parsed: Dict,
        champion_matches_a: Dict[str, int],
        champion_matches_b: Dict[str, int],
    ) -> Tuple[float, bool]:
        """Calculate breeding score based on champion presence and inbreeding.
        
        Scoring logic:
        - Base: 10 points per champion match (weighted by champion_weights)
        - Bonus: 5 points per additional occurrence of same champion
        - Inbreeding penalty: -15 points per detected duplicate sire
        - Normalize to 0-100
        
        Parameters
        ----------
        parent_a_parsed : Dict
            Parsed lineage of parent A.
        parent_b_parsed : Dict
            Parsed lineage of parent B.
        champion_matches_a : Dict[str, int]
            Champion matches in parent A.
        champion_matches_b : Dict[str, int]
            Champion matches in parent B.
        
        Returns
        -------
        Tuple[float, bool]
            (normalized_score, inbreeding_risk)
        """
        score = 0.0
        inbreeding_risk = False

        # Champion presence scoring
        for champion, count in champion_matches_a.items():
            weight = self.champion_weights.get(champion, 1.0)
            score += 10 * weight  # Base points
            if count > 1:
                score += 5 * (count - 1) * weight  # Bonus for frequency

        for champion, count in champion_matches_b.items():
            weight = self.champion_weights.get(champion, 1.0)
            score += 10 * weight
            if count > 1:
                score += 5 * (count - 1) * weight

        # Inbreeding detection: check for duplicate sires
        sire_a = parent_a_parsed.get("sire", "").lower() if parent_a_parsed.get("sire") else ""
        sire_b = parent_b_parsed.get("sire", "").lower() if parent_b_parsed.get("sire") else ""

        if sire_a and sire_b and sire_a == sire_b:
            inbreeding_risk = True
            score -= self.inbreeding_penalty

        # Check for champion appearing in both parents (homozygous concentration)
        for champion in self.champions:
            if (champion.lower() in (parent_a_parsed.get("raw", "").lower())) and (
                champion.lower() in (parent_b_parsed.get("raw", "").lower())
            ):
                score += 5  # Bonus for homozygous champion concentration

        # Normalize to 0-100
        normalized_score = max(0, min(100, score))
        return normalized_score, inbreeding_risk

    def score_combination(self, parent_a: str, parent_b: str) -> ScoringResult:
        """Score a proposed breeding combination.
        
        Parameters
        ----------
        parent_a : str
            Lineage of parent A, e.g., "498 x King George".
        parent_b : str
            Lineage of parent B, e.g., "1310 x Arasunu".
        
        Returns
        -------
        ScoringResult
            Structured result with score, explanation, matches, and metadata.
        """
        # Parse lineages
        parent_a_parsed = self.parse_lineage(parent_a)
        parent_b_parsed = self.parse_lineage(parent_b)

        # Detect champions
        matches_a = self.detect_champions(parent_a_parsed)
        matches_b = self.detect_champions(parent_b_parsed)
        all_matches = {**matches_a}
        for c, count in matches_b.items():
            all_matches[c] = all_matches.get(c, 0) + count

        # Calculate score
        score, inbreeding_risk = self.calculate_score(
            parent_a_parsed, parent_b_parsed, matches_a, matches_b
        )

        # Generate explanation
        explanation = self._generate_explanation(
            parent_a, parent_b, all_matches, inbreeding_risk, score
        )

        result = ScoringResult(
            score=score,
            explanation=explanation,
            champion_matches=all_matches,
            inbreeding_risk=inbreeding_risk,
            parent_a_parsed=parent_a_parsed,
            parent_b_parsed=parent_b_parsed,
        )

        logger.info(f"Scored combination: {parent_a} x {parent_b} = {score:.1f}")
        return result

    def _generate_explanation(
        self, parent_a: str, parent_b: str, matches: Dict[str, int], inbreeding: bool, score: float
    ) -> str:
        """Generate human-readable explanation of the score."""
        lines = [
            f"Combination: {parent_a} × {parent_b}",
            f"Score: {score:.1f} / 100",
        ]

        if matches:
            champion_str = ", ".join([f"{c} ({n})" for c, n in sorted(matches.items())])
            lines.append(f"Champion genetics detected: {champion_str}")
        else:
            lines.append("No known champion genetics detected in this cross.")

        if inbreeding:
            lines.append("⚠️  Inbreeding risk detected (duplicate sires).")
        else:
            lines.append("Genetic diversity appears healthy.")

        if score >= 80:
            lines.append("✓ Excellent genetic combination.")
        elif score >= 60:
            lines.append("✓ Good genetic foundation.")
        elif score >= 40:
            lines.append("~ Moderate genetic profile.")
        else:
            lines.append("⚠️  Limited champion genetics in this pairing.")

        return "\n".join(lines)

    def bulk_evaluate(self, combinations: List[Tuple[str, str]]) -> List[ScoringResult]:
        """Score multiple breeding combinations.
        
        Parameters
        ----------
        combinations : List[Tuple[str, str]]
            List of (parent_a, parent_b) tuples.
        
        Returns
        -------
        List[ScoringResult]
            Scores for each combination.
        """
        results = []
        for parent_a, parent_b in combinations:
            result = self.score_combination(parent_a, parent_b)
            results.append(result)
        return results


# Try importing pandas (optional for convenience)
try:
    import pandas as pd
except ImportError:
    pd = None
