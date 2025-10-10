"""Diagnostic tools for IdeaRank analysis.

Helps identify common issues like flat scores, missing data, etc.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class IdeaRankDiagnostics:
    """Diagnostic tools for IdeaRank data quality."""
    
    def __init__(self, db_path: str):
        """Initialize diagnostics with database path.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        logger.info(f"Initialized diagnostics for: {db_path}")
    
    def _fetch_data(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic checks.
        
        Returns:
            Dictionary with diagnostic results and warnings
        """
        results = {
            'issues': [],
            'warnings': [],
            'info': [],
            'checks_passed': 0,
            'checks_failed': 0,
        }
        
        logger.info("Running full diagnostics...")
        
        # Check 1: Flat Cohesion
        cohesion_check = self.check_flat_cohesion()
        if cohesion_check['is_flat']:
            results['issues'].append({
                'type': 'flat_cohesion',
                'severity': 'high',
                'message': cohesion_check['message'],
                'fix': 'Normalize topic weights, recompute per-item entropy'
            })
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1
        
        # Check 2: Flat Trust
        trust_check = self.check_flat_trust()
        if trust_check['is_flat']:
            results['issues'].append({
                'type': 'flat_trust',
                'severity': 'medium',
                'message': trust_check['message'],
                'fix': 'Build Trust pipeline (citations, timestamps, correction rate)'
            })
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1
        
        # Check 3: Learning Score Issues
        learning_check = self.check_learning_issues()
        if learning_check['has_issues']:
            results['warnings'].append({
                'type': 'learning_decay',
                'severity': 'medium',
                'message': learning_check['message'],
                'fix': 'Verify nearest_prior_self() logic - compare to nearest prior in semantic space'
            })
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1
        
        # Check 4: Missing Timestamps
        timestamp_check = self.check_missing_timestamps()
        if timestamp_check['missing_count'] > 0:
            results['warnings'].append({
                'type': 'missing_timestamps',
                'severity': 'low',
                'message': timestamp_check['message'],
                'fix': 'Use upload dates from YouTube API or Ghost published_at'
            })
        
        # Check 5: Narrow Score Spread
        spread_check = self.check_score_spread()
        if spread_check['is_narrow']:
            results['warnings'].append({
                'type': 'narrow_spread',
                'severity': 'medium',
                'message': spread_check['message'],
                'fix': 'Enable C & T variability, add noise floor to avoid identical values'
            })
            results['checks_failed'] += 1
        else:
            results['checks_passed'] += 1
        
        # Summary stats
        results['info'].append(f"Total content pieces: {self.get_total_count()}")
        results['info'].append(f"Content sources: {self.get_source_count()}")
        results['info'].append(f"Date range: {self.get_date_range()}")
        
        logger.info(f"Diagnostics complete: {results['checks_passed']} passed, {results['checks_failed']} failed")
        
        return results
    
    def check_flat_cohesion(self) -> Dict[str, Any]:
        """Check for flat cohesion scores (constant or near-constant).
        
        Returns:
            Dictionary with is_flat, variance, message
        """
        query = """
            SELECT cohesion_score
            FROM idearank_scores
            WHERE cohesion_score IS NOT NULL
        """
        data = self._fetch_data(query)
        
        if not data:
            return {'is_flat': False, 'variance': 0, 'message': 'No cohesion scores found'}
        
        scores = [row['cohesion_score'] for row in data]
        
        if not NUMPY_AVAILABLE:
            # Simple variance check
            mean = sum(scores) / len(scores)
            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        else:
            variance = np.var(scores)
        
        is_flat = variance < 0.01  # Very low variance indicates flat scores
        
        if is_flat:
            message = f"⚠️ Cohesion scores are flat (variance: {variance:.6f}). Likely using static defaults or degenerate topic vectors."
        else:
            message = f"✓ Cohesion has healthy variance ({variance:.4f})"
        
        return {
            'is_flat': is_flat,
            'variance': variance,
            'mean': sum(scores) / len(scores),
            'message': message
        }
    
    def check_flat_trust(self) -> Dict[str, Any]:
        """Check for flat trust scores.
        
        Returns:
            Dictionary with is_flat, common_value, message
        """
        query = """
            SELECT trust_score, COUNT(*) as count
            FROM idearank_scores
            WHERE trust_score IS NOT NULL
            GROUP BY trust_score
            ORDER BY count DESC
            LIMIT 1
        """
        data = self._fetch_data(query)
        
        if not data:
            return {'is_flat': False, 'message': 'No trust scores found'}
        
        most_common = data[0]
        total_query = "SELECT COUNT(*) as total FROM idearank_scores WHERE trust_score IS NOT NULL"
        total = self._fetch_data(total_query)[0]['total']
        
        percentage = (most_common['count'] / total) * 100
        
        is_flat = percentage > 80  # More than 80% have same value
        
        common_value = most_common['trust_score']
        
        # Check for specific hardcoded values
        if abs(common_value - 0.45) < 0.01 or abs(common_value - 0.5) < 0.01:
            message = f"⚠️ Trust scores are flat ({percentage:.1f}% are {common_value:.2f}). Likely using hardcoded fallback value."
        elif is_flat:
            message = f"⚠️ Trust scores lack variability ({percentage:.1f}% are {common_value:.2f})"
        else:
            message = f"✓ Trust has healthy distribution"
        
        return {
            'is_flat': is_flat,
            'common_value': common_value,
            'percentage': percentage,
            'message': message
        }
    
    def check_learning_issues(self) -> Dict[str, Any]:
        """Check for learning score issues (linear decay, etc.).
        
        Returns:
            Dictionary with has_issues, trend, message
        """
        query = """
            SELECT c.published_at, s.learning_score
            FROM content_items c
            JOIN idearank_scores s ON c.id = s.content_item_id
            WHERE s.learning_score IS NOT NULL
            ORDER BY c.published_at
        """
        data = self._fetch_data(query)
        
        if len(data) < 3:
            return {'has_issues': False, 'message': 'Not enough data'}
        
        scores = [row['learning_score'] for row in data]
        
        # Check for linear decay (consistent decrease)
        if not NUMPY_AVAILABLE:
            # Simple check: are scores mostly decreasing?
            decreases = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i-1])
            is_decreasing = (decreases / (len(scores) - 1)) > 0.7
            trend = "decreasing" if is_decreasing else "mixed"
        else:
            # Linear regression to detect trend
            x = np.arange(len(scores))
            coeffs = np.polyfit(x, scores, 1)
            slope = coeffs[0]
            trend = "decreasing" if slope < -0.01 else "increasing" if slope > 0.01 else "flat"
        
        has_issues = trend == "decreasing"
        
        if has_issues:
            message = f"⚠️ Learning scores show linear decay ({trend}). Check nearest_prior_self() logic - should compare to nearest prior in semantic space, not time-based averaging."
        else:
            message = f"✓ Learning scores show healthy progression ({trend})"
        
        return {
            'has_issues': has_issues,
            'trend': trend,
            'message': message
        }
    
    def check_missing_timestamps(self) -> Dict[str, Any]:
        """Check for missing or invalid timestamps.
        
        Returns:
            Dictionary with missing_count, message
        """
        query = """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END) as missing
            FROM content_items
        """
        data = self._fetch_data(query)[0]
        
        missing_count = data['missing']
        total = data['total']
        percentage = (missing_count / total * 100) if total > 0 else 0
        
        if missing_count > 0:
            message = f"⚠️ {missing_count}/{total} content items ({percentage:.1f}%) have missing timestamps"
        else:
            message = f"✓ All content items have timestamps"
        
        return {
            'missing_count': missing_count,
            'total': total,
            'percentage': percentage,
            'message': message
        }
    
    def check_score_spread(self) -> Dict[str, Any]:
        """Check if scores have healthy spread or are clustered.
        
        Returns:
            Dictionary with is_narrow, std_dev, message
        """
        query = """
            SELECT score
            FROM idearank_scores
            WHERE score IS NOT NULL
        """
        data = self._fetch_data(query)
        
        if not data:
            return {'is_narrow': False, 'message': 'No scores found'}
        
        scores = [row['score'] for row in data]
        
        if not NUMPY_AVAILABLE:
            mean = sum(scores) / len(scores)
            std_dev = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        else:
            std_dev = np.std(scores)
        
        is_narrow = std_dev < 0.05  # Very tight clustering
        
        if is_narrow:
            message = f"⚠️ Scores have narrow spread (std: {std_dev:.4f}). Enable C & T variability, add noise floor."
        else:
            message = f"✓ Scores have healthy spread (std: {std_dev:.4f})"
        
        return {
            'is_narrow': is_narrow,
            'std_dev': std_dev,
            'mean': sum(scores) / len(scores),
            'message': message
        }
    
    def get_factor_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each factor.
        
        Returns:
            Dictionary with stats for U, C, L, Q, T
        """
        query = """
            SELECT 
                AVG(uniqueness_score) as u_mean,
                AVG(cohesion_score) as c_mean,
                AVG(learning_score) as l_mean,
                AVG(quality_score) as q_mean,
                AVG(trust_score) as t_mean
            FROM idearank_scores
        """
        means = self._fetch_data(query)[0]
        
        # Get variance for each factor
        factors = {}
        for factor_name, col_name in [
            ('Uniqueness', 'uniqueness_score'),
            ('Cohesion', 'cohesion_score'),
            ('Learning', 'learning_score'),
            ('Quality', 'quality_score'),
            ('Trust', 'trust_score'),
        ]:
            query = f"SELECT {col_name} FROM idearank_scores WHERE {col_name} IS NOT NULL"
            scores = [row[col_name] for row in self._fetch_data(query)]
            
            if scores:
                mean = sum(scores) / len(scores)
                if NUMPY_AVAILABLE:
                    std = np.std(scores)
                    variance = np.var(scores)
                else:
                    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
                    std = variance ** 0.5
                
                factors[factor_name] = {
                    'mean': mean,
                    'std': std,
                    'variance': variance,
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        return factors
    
    def get_total_count(self) -> int:
        """Get total number of content items."""
        query = "SELECT COUNT(*) as count FROM content_items"
        return self._fetch_data(query)[0]['count']
    
    def get_source_count(self) -> int:
        """Get number of unique content sources."""
        query = "SELECT COUNT(DISTINCT content_source_id) as count FROM content_items"
        return self._fetch_data(query)[0]['count']
    
    def get_date_range(self) -> str:
        """Get date range of content."""
        query = """
            SELECT 
                MIN(published_at) as min_date,
                MAX(published_at) as max_date
            FROM content_items
            WHERE published_at IS NOT NULL
        """
        data = self._fetch_data(query)[0]
        
        if data['min_date'] and data['max_date']:
            return f"{data['min_date'][:10]} to {data['max_date'][:10]}"
        return "Unknown"
    
    def print_report(self) -> None:
        """Print comprehensive diagnostic report."""
        print("\n" + "=" * 70)
        print("IdeaRank Diagnostics Report")
        print("=" * 70)
        
        results = self.run_full_diagnostics()
        
        # Summary
        print(f"\n📊 Summary:")
        for info in results['info']:
            print(f"  {info}")
        
        # Checks passed
        print(f"\n✅ Checks Passed: {results['checks_passed']}")
        print(f"❌ Checks Failed: {results['checks_failed']}")
        
        # Issues
        if results['issues']:
            print(f"\n🔴 Issues Found ({len(results['issues'])}):")
            for i, issue in enumerate(results['issues'], 1):
                print(f"\n  {i}. {issue['type'].upper()} (Severity: {issue['severity']})")
                print(f"     {issue['message']}")
                print(f"     Fix: {issue['fix']}")
        else:
            print("\n✅ No critical issues found")
        
        # Warnings
        if results['warnings']:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for i, warning in enumerate(results['warnings'], 1):
                print(f"\n  {i}. {warning['type'].upper()}")
                print(f"     {warning['message']}")
                print(f"     Fix: {warning['fix']}")
        
        # Factor statistics
        print(f"\n📈 Factor Statistics:")
        print("-" * 70)
        factors = self.get_factor_stats()
        
        print(f"\n{'Factor':<12} {'Mean':>8} {'Std Dev':>10} {'Min':>8} {'Max':>8} {'Status'}")
        print("-" * 70)
        
        for name, stats in factors.items():
            # Determine status
            if stats['variance'] < 0.01:
                status = "⚠️ FLAT"
            elif stats['std'] < 0.05:
                status = "⚠️ Low Var"
            else:
                status = "✓ Healthy"
            
            print(f"{name:<12} {stats['mean']:>8.4f} {stats['std']:>10.4f} "
                  f"{stats['min']:>8.4f} {stats['max']:>8.4f} {status}")
        
        print("\n" + "=" * 70)
        
        # Recommendations
        if results['issues'] or results['warnings']:
            print("\n💡 Recommendations:")
            print("-" * 70)
            
            issue_types = [i['type'] for i in results['issues']]
            warning_types = [w['type'] for w in results['warnings']]
            
            if 'flat_cohesion' in issue_types:
                print("  1. Check topic model: Print H(topic_distribution) per content item")
                print("     Normalize topic weights to ensure entropy varies")
            
            if 'flat_trust' in issue_types:
                print("  2. Implement Trust pipeline:")
                print("     - Parse citations from content")
                print("     - Track timestamps and corrections")
                print("     - Calculate source diversity")
            
            if 'learning_decay' in warning_types:
                print("  3. Fix Learning calculation:")
                print("     - Verify nearest_prior_self() finds semantic nearest neighbor")
                print("     - Don't use time-based windows for averaging")
            
            if 'narrow_spread' in warning_types:
                print("  4. Increase score diversity:")
                print("     - Enable Cohesion and Trust variability")
                print("     - Add small noise floor (ε = 0.001) to avoid identical values")
            
            print()
        
        print("=" * 70)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_diagnostic_report(db_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Create a diagnostic report for an IdeaRank database.
    
    Args:
        db_path: Path to database
        output_path: Optional path to save report as JSON
        
    Returns:
        Dictionary with diagnostic results
    """
    with IdeaRankDiagnostics(db_path) as diag:
        results = diag.run_full_diagnostics()
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved diagnostic report: {output_path}")
        
        return results

