import pandas as pd
import numpy as np
from datetime import datetime

# ==================== CONFIGURATION ====================
class Config:
    ERROR_THRESHOLDS = {
        'total_delta_negative': -1.0,
        'low_confidence': 0.6,
        'q_value_inconsistency': 0.5,
        'msx_empty': 0
    }
    
    CONFIDENCE_WEIGHTS = {
        'total_delta': 0.3,
        'q_value_consistency': 0.4,
        'msx_clarity': 0.3
    }
    
    ACTION_MAP = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    COMPONENT_MAP = {'turn': 'Turn', 'goal': 'Goal', 'blocked': 'Blocked', 'safe': 'Safe'}
    
    DATA_FILE = "excel-results/dataset-final.xlsx"
    OUTPUT_FILE = "excel-results/validation-report-final.xlsx"
    REQUIRED_CONFIDENCE = 0.7

# ==================== DATA LOADER ====================
class DataLoader:
    def __init__(self, data_file=Config.DATA_FILE):
        self.data_file = data_file
        self.episode_log = None
        self.q_tables = {}
        
    def load_data(self):
        try:
            self.episode_log = pd.read_excel(self.data_file, sheet_name='Episode Log')
            q_sheets = ['Q_turn', 'Q_goal', 'Q_blocked', 'Q_safe', 'Q_Total']
            for sheet in q_sheets:
                self.q_tables[sheet] = pd.read_excel(self.data_file, sheet_name=sheet)
            
            print(f"Loaded {len(self.episode_log)} episodes and {len(self.q_tables)} Q-tables")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_q_value(self, state, action, q_table_name):
        try:
            q_table = self.q_tables[q_table_name]
            state_data = q_table[q_table['State'] == state]
            if len(state_data) == 0:
                return None
            action_columns = ['Q_Up', 'Q_Down', 'Q_Left', 'Q_Right']
            return state_data[action_columns[action]].values[0]
        except:
            return None
    
    def get_all_q_values(self, state):
        result = {}
        for q_table_name in self.q_tables.keys():
            result[q_table_name] = {}
            for action in range(4):
                result[q_table_name][action] = self.get_q_value(state, action, q_table_name)
        return result

# ==================== ERROR DETECTOR ====================
class ErrorDetector:
    def __init__(self, config=Config):
        self.config = config
        
    def detect_potential_errors(self, episode_log):
        errors = []
        for idx, row in episode_log.iterrows():
            errors.extend(self._analyze_single_decision(row))
        return errors
    
    def _analyze_single_decision(self, row):
        errors = []
        state = row['State']
        chosen = row['Chosen Action']
        alternative = row['Alternative Action']
        
        # 1. Check total delta < 0
        if row['Total Δ'] < 0:
            errors.append({
                'state': state, 'chosen_action': chosen, 'alternative_action': alternative,
                'error_type': 'NEGATIVE_TOTAL_DELTA', 'severity': 'HIGH',
                'description': f'Total Δ negative ({row["Total Δ"]:.2f})', 'value': row['Total Δ']
            })
        
        # 2. Check Q_value consistency
        q_consistency_error = self._check_q_value_consistency(row)
        if q_consistency_error:
            errors.append(q_consistency_error)
        
        # 3. Check component with negative delta but in MSX+
        negative_in_plus = self._check_negative_delta_in_msx_plus(row)
        errors.extend(negative_in_plus)
        
        # 4. Check component with positive delta but in MSX-
        positive_in_minus = self._check_positive_delta_in_msx_minus(row)
        errors.extend(positive_in_minus)
        
        # 5. Check empty MSX+
        if len(row['MSX+']) == 0:
            errors.append({
                'state': state, 'chosen_action': chosen, 'alternative_action': alternative,
                'error_type': 'EMPTY_MSX_PLUS', 'severity': 'MEDIUM',
                'description': 'Empty MSX+ - no positive factors', 'value': 0
            })
        
        return errors
    
    def _check_q_value_consistency(self, row):
        """Check Q-value consistency"""
        state = row['State']
        chosen = row['Chosen Action']
        alternative = row['Alternative Action']
        
        inconsistencies = []
        
        # Check if sum of component deltas matches Total Δ
        component_sum = row['Δ_turn'] + row['Δ_goal'] + row['Δ_blocked'] + row['Δ_safe']
        total_delta = row['Total Δ']
        
        if abs(component_sum - total_delta) > 0.01:  # Allow small error margin
            inconsistencies.append(f"Sum of component Δ ({component_sum:.2f}) ≠ Total Δ ({total_delta:.2f})")
        
        if inconsistencies:
            return {
                'state': state, 'chosen_action': chosen, 'alternative_action': alternative,
                'error_type': 'Q_VALUE_INCONSISTENCY', 'severity': 'MEDIUM',
                'description': f'Q-value inconsistencies: {", ".join(inconsistencies)}',
                'value': len(inconsistencies)
            }
        return None
    
    def _check_negative_delta_in_msx_plus(self, row):
        """Check components with negative delta but in MSX+"""
        state = row['State']
        chosen = row['Chosen Action']
        alternative = row['Alternative Action']
        
        errors = []
        for component in row['MSX+']:
            delta_value = getattr(row, f'Δ_{component}', 0)
            if delta_value < 0:
                errors.append({
                    'state': state, 'chosen_action': chosen, 'alternative_action': alternative,
                    'error_type': 'NEGATIVE_DELTA_IN_MSX_PLUS', 'severity': 'MEDIUM',
                    'description': f'Component {component} has negative Δ ({delta_value:.2f}) but is in MSX+',
                    'value': delta_value
                })
        return errors
    
    def _check_positive_delta_in_msx_minus(self, row):
        """Check components with positive delta but in MSX-"""
        state = row['State']
        chosen = row['Chosen Action']
        alternative = row['Alternative Action']
        
        errors = []
        for component in row['MSX-']:
            delta_value = getattr(row, f'Δ_{component}', 0)
            if delta_value > 0:
                errors.append({
                    'state': state, 'chosen_action': chosen, 'alternative_action': alternative,
                    'error_type': 'POSITIVE_DELTA_IN_MSX_MINUS', 'severity': 'MEDIUM',
                    'description': f'Component {component} has positive Δ ({delta_value:.2f}) but is in MSX-',
                    'value': delta_value
                })
        return errors

class ExplanationGenerator:
    def __init__(self, config=Config):
        self.config = config
        
    def generate_explanation(self, row, q_values=None):
        state = row['State']
        chosen = row['Chosen Action']
        alternative = row['Alternative Action']
        
        explanation = {
            'state': state,
            'chosen_action': chosen,
            'alternative_action': alternative,
            'chosen_action_name': self.config.ACTION_MAP[chosen],
            'alternative_action_name': self.config.ACTION_MAP[alternative],
            'explanation_text': self._create_detailed_explanation(row),
            'reasoning_factors': self._extract_reasoning_factors(row),
            'key_factors': self._identify_key_factors(row)
        }
        
        if q_values:
            explanation['q_value_analysis'] = self._analyze_q_values(row, q_values)
        
        return explanation
    
    def _create_detailed_explanation(self, row):
        state = row['State']
        chosen = self.config.ACTION_MAP[row['Chosen Action']]
        alternative = self.config.ACTION_MAP[row['Alternative Action']]
        
        # Analyze influencing factors
        positive_factors = []
        negative_factors = []
        
        # Classify positive and negative factors
        for component in ['turn', 'goal', 'blocked', 'safe']:
            delta = getattr(row, f'Δ_{component}')
            component_name = self.config.COMPONENT_MAP[component]
            
            if component in row['MSX+']:
                if component == 'goal' and delta > 0:
                    positive_factors.append(f"Move closer to goal ({delta:+.2f})")
                elif component == 'safe' and delta > 0:
                    positive_factors.append(f"Safer move ({delta:+.2f})")
                elif component == 'blocked' and delta > 0:
                    positive_factors.append(f"Better avoid blocked positions ({delta:+.2f})")
                elif component == 'turn' and delta > 0:
                    positive_factors.append(f"More efficient movement ({delta:+.2f})")
            
            if component in row['MSX-']:
                if component == 'turn' and delta < 0:
                    negative_factors.append(f"Requires more turning ({delta:+.2f})")
                elif component == 'safe' and delta < 0:
                    negative_factors.append(f"Less safe ({delta:+.2f})")
                elif component == 'goal' and delta < 0:
                    negative_factors.append(f"Further from goal ({delta:+.2f})")
        
        # Build natural language explanation
        explanation = f"At position {state}, the system decides to move {chosen} instead of {alternative}.\n\n"
        
        # Positive factors
        if positive_factors:
            explanation += "This choice provides the following benefits:\n"
            for i, factor in enumerate(positive_factors, 1):
                explanation += f"• {factor}\n"
        
        # Negative factors (but acceptable)
        if negative_factors:
            explanation += "\nHowever, there are some limitations that must be accepted:\n"
            for i, factor in enumerate(negative_factors, 1):
                explanation += f"• {factor}\n"
        
        # Summary
        total_delta = row['Total Δ']
        if total_delta > 2:
            conclusion = "This is a very good choice"
        elif total_delta > 1:
            conclusion = "This is a fairly good choice"
        elif total_delta > 0:
            conclusion = "This is a beneficial choice"
        else:
            conclusion = "This choice needs reconsideration"
        
        explanation += f"\n{conclusion} with total improvement value of {total_delta:.2f}."
        
        # Add main reason if available
        primary_reason = self._get_primary_reason(row)
        if primary_reason:
            explanation += f" The main reason is {primary_reason}."
        
        return explanation
    
    def _extract_reasoning_factors(self, row):
        factors = []
        for component in row['MSX+']:
            component_name = self.config.COMPONENT_MAP.get(component, component)
            delta_value = getattr(row, f'Δ_{component}', 0)
            
            if component == 'goal':
                desc = f"Move closer to goal ({delta_value:+.2f})"
            elif component == 'safe':
                desc = f"Increase safety ({delta_value:+.2f})"
            elif component == 'blocked':
                desc = f"Reduce blocking risk ({delta_value:+.2f})"
            elif component == 'turn':
                desc = f"Optimize movement ({delta_value:+.2f})"
            else:
                desc = f"Improve {component_name.lower()} ({delta_value:+.2f})"
                
            factors.append({
                'component': component, 'component_name': component_name, 'impact': 'positive',
                'value': delta_value, 'description': desc
            })
            
        for component in row['MSX-']:
            component_name = self.config.COMPONENT_MAP.get(component, component)
            delta_value = getattr(row, f'Δ_{component}', 0)
            
            if component == 'turn':
                desc = f"Accept more turning ({delta_value:+.2f})"
            elif component == 'safe':
                desc = f"Accept some risk ({delta_value:+.2f})"
            elif component == 'goal':
                desc = f"Accept being further from goal ({delta_value:+.2f})"
            else:
                desc = f"Accept reduced {component_name.lower()} ({delta_value:+.2f})"
                
            factors.append({
                'component': component, 'component_name': component_name, 'impact': 'negative',
                'value': delta_value, 'description': desc
            })
            
        return factors
    
    def _identify_key_factors(self, row):
        key_factors = []
        
        # Find most influential factor
        deltas = {
            'turn': abs(row['Δ_turn']), 
            'goal': abs(row['Δ_goal']), 
            'blocked': abs(row['Δ_blocked']), 
            'safe': abs(row['Δ_safe'])
        }
        
        # Add factors with significant delta
        for component, delta in deltas.items():
            if delta > 1.0:  # Significant impact threshold
                component_name = self.config.COMPONENT_MAP[component]
                actual_delta = getattr(row, f'Δ_{component}')
                
                if actual_delta > 0:
                    key_factors.append(f"{component_name} (improvement {actual_delta:.2f})")
                else:
                    key_factors.append(f"{component_name} (reduction {abs(actual_delta):.2f})")
        
        return key_factors
    
    def _get_primary_reason(self, row):
        """Determine main reason for decision"""
        # Find most influential factor
        components = ['goal', 'safe', 'blocked', 'turn']
        max_impact = 0
        primary_component = None
        
        for component in components:
            delta = abs(getattr(row, f'Δ_{component}'))
            if delta > max_impact and component in row['MSX+']:
                max_impact = delta
                primary_component = component
        
        # If no factor found in MSX+, find factor with highest delta
        if primary_component is None:
            for component in components:
                delta = abs(getattr(row, f'Δ_{component}'))
                if delta > max_impact:
                    max_impact = delta
                    primary_component = component
        
        # Map to specific reason
        if primary_component == 'goal' and row['Δ_goal'] > 0:
            return "prioritizing moving closer to goal"
        elif primary_component == 'safe' and row['Δ_safe'] > 0:
            return "ensuring move safety"
        elif primary_component == 'blocked' and row['Δ_blocked'] > 0:
            return "avoiding blocked positions"
        elif primary_component == 'turn' and row['Δ_turn'] > 0:
            return "optimizing movement turns"
        elif primary_component == 'turn' and row['Δ_turn'] < 0:
            return "accepting more turning for greater benefit"
        elif primary_component == 'safe' and row['Δ_safe'] < 0:
            return "accepting risk for faster progress"
        
        # Special case when no factor stands out
        if row['Total Δ'] > 1.5:
            # Analyze overall factors
            positive_count = len(row['MSX+'])
            if positive_count >= 3:
                return "majority of factors are improved"
            elif positive_count == 2:
                factors = [self.config.COMPONENT_MAP[comp] for comp in row['MSX+']]
                return f"improving both {factors[0]} and {factors[1]}"
        
        # Default only when truly balanced
        if len(row['MSX+']) == len(row['MSX-']) == 2:
            return "balancing benefits and limitations"
        
        return "multi-objective optimization"
    
    def _analyze_q_values(self, row, q_values):
        analysis = {}
        for q_table_name, values in q_values.items():
            chosen_q = values.get(row['Chosen Action'])
            alternative_q = values.get(row['Alternative Action'])
            if chosen_q is not None and alternative_q is not None:
                advantage = chosen_q - alternative_q
                analysis[q_table_name] = {
                    'chosen_q': chosen_q, 'alternative_q': alternative_q,
                    'advantage': advantage, 'preferred': advantage > 0
                }
        return analysis


# ==================== EXPLANATION VALIDATOR ====================
class ExplanationValidator:
    def __init__(self, data_loader, config=Config):
        self.data_loader = data_loader
        self.config = config
        
    def validate_explanation(self, row, explanation):
        state = row['State']
        chosen = row['Chosen Action']
        
        validation_result = {
            'state': state, 'chosen_action': chosen, 'validation_passed': True,
            'validation_checks': [], 'failed_checks': [], 'warnings': [], 'overall_score': 0
        }
        
        q_values = self.data_loader.get_all_q_values(state)
        checks = [
            self._check_q_value_consistency(row, q_values),
            self._check_msx_alignment(row, q_values),
            self._check_delta_consistency(row),
            self._check_action_optimality(row, q_values)
        ]
        
        passed_checks = 0
        for check in checks:
            validation_result['validation_checks'].append(check)
            if check['passed']:
                passed_checks += 1
            else:
                validation_result['failed_checks'].append(check)
                validation_result['validation_passed'] = False
        
        total_checks = len(checks)
        validation_result['overall_score'] = passed_checks / total_checks if total_checks > 0 else 0
        validation_result['warnings'] = self._generate_warnings(row, q_values)
        
        return validation_result
    
    def _check_q_value_consistency(self, row, q_values):
        state = row['State']
        chosen = row['Chosen Action']
        alternative = row['Alternative Action']
        
        check = {
            'name': 'Q_VALUE_CONSISTENCY',
            'description': 'Chosen action has higher Q-value than alternative',
            'passed': True, 'details': {}
        }
        
        q_total = q_values.get('Q_Total', {})
        chosen_q = q_total.get(chosen)
        alternative_q = q_total.get(alternative)
        
        if chosen_q is not None and alternative_q is not None:
            check['details'] = {
                'chosen_q': chosen_q, 'alternative_q': alternative_q,
                'difference': chosen_q - alternative_q
            }
            if chosen_q < alternative_q:
                check['passed'] = False
                check['details']['issue'] = 'Chosen action has lower Q-value than alternative'
        
        return check
    
    def _check_msx_alignment(self, row, q_values):
        check = {
            'name': 'MSX_ALIGNMENT',
            'description': 'MSX+ components align with Q-values',
            'passed': True, 'details': {}
        }
        
        msx_plus = row['MSX+']
        alignment_issues = []
        
        for component in ['goal', 'safe', 'turn', 'blocked']:
            q_table_name = f'Q_{component}'
            if q_table_name in q_values:
                chosen = row['Chosen Action']
                chosen_q = q_values[q_table_name].get(chosen)
                
                if component in msx_plus and chosen_q is not None:
                    all_q_values = list(q_values[q_table_name].values())
                    max_q = max([v for v in all_q_values if v is not None])
                    
                    if chosen_q < max_q * 0.8:
                        alignment_issues.append(f"{component}: Q-value ({chosen_q:.2f}) low compared to max ({max_q:.2f})")
        
        if alignment_issues:
            check['passed'] = False
            check['details']['issues'] = alignment_issues
        
        return check
    
    def _check_delta_consistency(self, row):
        check = {
            'name': 'DELTA_CONSISTENCY',
            'description': 'Delta values are consistent',
            'passed': True, 'details': {}
        }
        
        component_sum = row['Δ_turn'] + row['Δ_goal'] + row['Δ_blocked'] + row['Δ_safe']
        total_delta = row['Total Δ']
        tolerance = 0.01
        
        if abs(component_sum - total_delta) > tolerance:
            check['passed'] = False
            check['details'] = {
                'component_sum': component_sum, 'total_delta': total_delta,
                'difference': component_sum - total_delta,
                'issue': 'Sum of component deltas does not match Total Δ'
            }
        
        return check
    
    def _check_action_optimality(self, row, q_values):
        check = {
            'name': 'ACTION_OPTIMALITY',
            'description': 'Chosen action is optimal according to Q-values',
            'passed': True, 'details': {}
        }
        
        q_total = q_values.get('Q_Total', {})
        if not q_total:
            return check
        
        valid_q_values = {k: v for k, v in q_total.items() if v is not None}
        if not valid_q_values:
            return check
        
        best_action = max(valid_q_values, key=valid_q_values.get)
        chosen = row['Chosen Action']
        
        check['details'] = {
            'chosen_action': chosen, 'best_action': best_action,
            'chosen_q': valid_q_values[chosen], 'best_q': valid_q_values[best_action]
        }
        
        if chosen != best_action:
            check['passed'] = False
            check['details']['issue'] = 'Chosen action is not the optimal one'
        
        return check
    
    def _generate_warnings(self, row, q_values):
        warnings = []
        if row['Total Δ'] < 0.5:
            warnings.append(f"Low Total Δ ({row['Total Δ']:.2f})")
        
        q_total = q_values.get('Q_Total', {})
        if q_total:
            q_values_list = [v for v in q_total.values() if v is not None]
            if q_values_list:
                q_range = max(q_values_list) - min(q_values_list)
                if q_range > 10.0:
                    warnings.append(f"Large Q-value range ({q_range:.2f})")
        
        return warnings

# ==================== CONFIDENCE SCORER ====================
class ConfidenceScorer:
    def __init__(self, config=Config):
        self.config = config
        
    def calculate_confidence(self, row, validation_result, q_values=None):
        confidence_factors = []
        
        total_delta_confidence = self._calculate_total_delta_confidence(row)
        confidence_factors.append(total_delta_confidence * self.config.CONFIDENCE_WEIGHTS['total_delta'])
        
        validation_confidence = validation_result.get('overall_score', 0)
        confidence_factors.append(validation_confidence * self.config.CONFIDENCE_WEIGHTS['q_value_consistency'])
        
        msx_clarity_confidence = self._calculate_msx_clarity_confidence(row)
        confidence_factors.append(msx_clarity_confidence * self.config.CONFIDENCE_WEIGHTS['msx_clarity'])
        
        if q_values:
            q_consistency_confidence = self._calculate_q_consistency_confidence(row, q_values)
            confidence_factors[1] += q_consistency_confidence * 0.2
        
        confidence_score = sum(confidence_factors)
        
        return {
            'confidence_score': max(0, min(1, confidence_score)),
            'confidence_level': self._get_confidence_level(confidence_score),
            'factor_breakdown': {
                'total_delta': total_delta_confidence,
                'validation': validation_confidence,
                'msx_clarity': msx_clarity_confidence
            }
        }
    
    def _calculate_total_delta_confidence(self, row):
        total_delta = row['Total Δ']
        normalized = (total_delta + 5) / 15
        return max(0, min(1, normalized))
    
    def _calculate_msx_clarity_confidence(self, row):
        msx_plus = row['MSX+']
        msx_minus = row['MSX-']
        defined_components = len(msx_plus) + len(msx_minus)
        clarity_ratio = defined_components / 4.0
        conflicting = set(msx_plus) & set(msx_minus)
        conflict_penalty = len(conflicting) * 0.2
        return max(0, clarity_ratio - conflict_penalty)
    
    def _calculate_q_consistency_confidence(self, row, q_values):
        q_total = q_values.get('Q_Total', {})
        chosen = row['Chosen Action']
        
        if not q_total or chosen not in q_total or q_total[chosen] is None:
            return 0.5
        
        valid_q_values = [v for v in q_total.values() if v is not None]
        if not valid_q_values:
            return 0.5
        
        max_q = max(valid_q_values)
        chosen_q = q_total[chosen]
        
        if max_q == 0:
            return 0.5
        
        ratio = chosen_q / max_q
        return max(0, min(1, ratio))
    
    def _get_confidence_level(self, score):
        if score >= 0.8: return "HIGH"
        elif score >= 0.6: return "MEDIUM"
        elif score >= 0.4: return "LOW"
        else: return "VERY_LOW"

# ==================== MAIN AUTO VALIDATION SYSTEM ====================
class AutoValidationSystem:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.error_detector = ErrorDetector()
        self.explanation_generator = ExplanationGenerator()
        self.explanation_validator = ExplanationValidator(self.data_loader)
        self.confidence_scorer = ConfidenceScorer()
        self.results = []
        
    def run_analysis(self):
        print("Starting Auto-Validation System...")
        
        if not self.data_loader.load_data():
            return False
        
        episode_log = self.data_loader.episode_log
        
        print("Detecting potential errors...")
        errors = self.error_detector.detect_potential_errors(episode_log)
        print(f"Detected {len(errors)} potential errors")
        
        print("Generating explanations and validation...")
        for idx, row in episode_log.iterrows():
            if idx % 10 == 0:
                print(f"Processing {idx}/{len(episode_log)}...")
            
            q_values = self.data_loader.get_all_q_values(row['State'])
            explanation = self.explanation_generator.generate_explanation(row, q_values)
            validation = self.explanation_validator.validate_explanation(row, explanation)
            confidence = self.confidence_scorer.calculate_confidence(row, validation, q_values)
            
            requires_review = (not validation['validation_passed'] or 
                             confidence['confidence_score'] < self.config.REQUIRED_CONFIDENCE)
            
            self.results.append({
                'row': row,
                'explanation': explanation,
                'validation': validation,
                'confidence': confidence,
                'requires_review': requires_review
            })
        
        print("Generating report...")
        self.generate_report()
        
        return True
    
    def generate_report(self):
        report_data = []
        
        for result in self.results:
            row = result['row']
            explanation = result['explanation']
            validation = result['validation']
            confidence = result['confidence']
            
            report_data.append({
                'State': row['State'],
                'Chosen_Action': explanation['chosen_action_name'],
                'Alternative_Action': explanation['alternative_action_name'],
                'Total_Delta': row['Total Δ'],
                'MSX_Plus': str(row['MSX+']),
                'MSX_Minus': str(row['MSX-']),
                'Validation_Passed': validation['validation_passed'],
                'Validation_Score': validation['overall_score'],
                'Confidence_Score': confidence['confidence_score'],
                'Confidence_Level': confidence['confidence_level'],
                'Requires_Review': result['requires_review'],
                'Explanation': explanation['explanation_text']
            })
        
        df_report = pd.DataFrame(report_data)
        
        with pd.ExcelWriter(self.config.OUTPUT_FILE, engine='openpyxl') as writer:
            df_report.to_excel(writer, sheet_name='Validation_Report', index=False)
            
            # Summary sheet
            summary_data = self._create_summary()
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Errors sheet
            errors = self.error_detector.detect_potential_errors(self.data_loader.episode_log)
            if errors:
                errors_df = pd.DataFrame(errors)
                errors_df.to_excel(writer, sheet_name='Detected_Errors', index=False)
        
        print(f"Report generated: {self.config.OUTPUT_FILE}")
        self._print_summary()
    
    def _create_summary(self):
        total_decisions = len(self.results)
        passed_validation = sum(1 for r in self.results if r['validation']['validation_passed'])
        high_confidence = sum(1 for r in self.results if r['confidence']['confidence_level'] == 'HIGH')
        needs_review = sum(1 for r in self.results if r['requires_review'])
        
        return {
            'Total_Decisions': total_decisions,
            'Decisions_Passed_Validation': passed_validation,
            'Validation_Success_Rate': f"{(passed_validation/total_decisions)*100:.1f}%",
            'High_Confidence_Decisions': high_confidence,
            'High_Confidence_Rate': f"{(high_confidence/total_decisions)*100:.1f}%",
            'Decisions_Needing_Review': needs_review,
            'Review_Rate': f"{(needs_review/total_decisions)*100:.1f}%",
            'Generated_At': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _print_summary(self):
        summary = self._create_summary()
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        print(f"Total decisions: {summary['Total_Decisions']}")
        print(f"Validation success rate: {summary['Validation_Success_Rate']}")
        print(f"High confidence rate: {summary['High_Confidence_Rate']}")
        print(f"Needs review: {summary['Decisions_Needing_Review']} ({summary['Review_Rate']})")
        print("="*50)

# ==================== RUN THE SYSTEM ====================
if __name__ == "__main__":
    system = AutoValidationSystem()
    success = system.run_analysis()
    
    if success:
        print("\nAuto-Validation completed!")
        print(f"Results saved to: {system.config.OUTPUT_FILE}")
    else:
        print("\nAuto-Validation failed!")