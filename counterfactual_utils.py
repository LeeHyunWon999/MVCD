import re


# ==============edit=========================
def _coerce_num(value):
    """Extract a leading signed int/float from a value that may carry trailing
    commentary, e.g. '4.0 (positive shift)' -> 4.0, '-1 (unchanged)' -> -1.
    Non-numeric text (Review/Image bodies) is returned unchanged. Newer models
    routinely append parenthetical rationale after the number, which the old
    float()/int() parse could not handle and left as a raw string, crashing the
    downstream `score.astype(float)`."""
    m = re.match(r'\s*([-+]?\d+(?:\.\d+)?)', str(value))
    if not m:
        return value
    num = m.group(1)
    return float(num) if '.' in num else int(num)
# ===========================================

def parse_counterfactual_scenarios(response_text, factor_names=None):
    scenarios = {}
    current_scenario = None
    current_sample = None
    factor_names = factor_names or []

    for line in response_text.split('\n'):
        line = line.strip()

        if not line:
            continue

        scenario_match = re.match(r'## Counterfactual Scenario \d+: (.*)', line)
        if scenario_match:
            scenario_name = scenario_match.group(1).strip()
            scenarios[scenario_name] = {"samples": {}}
            current_scenario = scenario_name
            continue

        sample_match = re.match(r'\*\*Sample (\d+)', line)
        if sample_match and current_scenario:
            sample_num = sample_match.group(1)
            sample_id = sample_num
            scenarios[current_scenario]["samples"][sample_id] = {}
            current_sample = sample_id
            continue

        if line.startswith('- ') and current_scenario and current_sample:
            content = line[2:].strip()
            # Some models wrap labels/values in markdown bold, e.g.
            # '- **Image**: N/A' or '- **Factor 1: 1** (note)'. Strip the bold
            # markers so keys match ('Image', not '**Image**'); otherwise the
            # required 'Image' key is absent and extraction yields 0 rows,
            # crashing refine() with KeyError: 'sample_id'.
            content = content.replace('**', '')

            value = None
            for factor in factor_names:
                if content.startswith(factor + ':'):
                    value = content[len(factor) + 1:].strip()
                    value = _coerce_num(value)          # [edit] leading-number extract
                    scenarios[current_scenario]["samples"][current_sample][factor] = value
                    break

            if value is None and ':' in content:        # [edit] None (not 0) = unmatched
                factor, value = map(str.strip, content.rsplit(':', 1))
                value = _coerce_num(value)              # [edit] leading-number extract
                scenarios[current_scenario]["samples"][current_sample][factor] = value

    return scenarios

def extract_counterfactual_data(response_text, factor_names=None):
    import pandas as pd

    scenarios = parse_counterfactual_scenarios(response_text, factor_names)

    all_samples = []

    for scenario_name, scenario_data in scenarios.items():
        for sample_id, sample_data in scenario_data["samples"].items():
            if "Image" in sample_data:
                sample_row = {
                    "scenario": scenario_name,
                    "sample_id": sample_id,
                    **sample_data
                }
                all_samples.append(sample_row)

    if all_samples:
        df = pd.DataFrame(all_samples)
        return scenarios, df

    return scenarios, pd.DataFrame()
