"""
Wolfram Language Theoretical Context Analysis Guide

This module provides code snippets to be added to the Wolfram Language notebook
for implementing theoretical context analysis to connect findings to established frameworks.
"""

# =============================================================================
# WOLFRAM LANGUAGE THEORETICAL CONTEXT CODE (Add to your notebook)
# =============================================================================

THEORETICAL_CONTEXT_CODE = '''
(* ===== Theoretical Context Analysis ===== *)

(* 1. Chew-Frautschi Expectations *)

(* Literature reference ranges (GeV⁻²) *)
literatureRanges = <|
  "meson" -> <|"range" -> {0.7, 1.1}, "typical" -> 0.9, 
               "reference" -> "Chew & Frautschi (1961), Donnachie & Landshoff (1992)"|>,
  "baryon" -> <|"range" -> {0.8, 1.2}, "typical" -> 1.0, 
               "reference" -> "Chew & Frautschi (1961), Capstick & Isgur (1986)"|>,
  "general" -> <|"range" -> {0.6, 1.3}, "typical" -> 0.9, 
                "reference" -> "General Regge phenomenology"|>
|>;

(* Extract fitted parameters *)
fittedAlphap = wlsFit["BestFitParameters"][[2]];
fittedAlphapErr = wlsFit["ParameterStandardErrors"][[2]];

(* Compare with baryon expectations *)
baryonRef = literatureRanges["baryon"];
expectedRange = baryonRef["range"];
typicalValue = baryonRef["typical"];

(* Calculate z-scores *)
zScoreToTypical = (fittedAlphap - typicalValue) / fittedAlphapErr;
zScoreToMin = (fittedAlphap - expectedRange[[1]]) / fittedAlphapErr;
zScoreToMax = (fittedAlphap - expectedRange[[2]]) / fittedAlphapErr;

(* Check if within expected range *)
withinRange = expectedRange[[1]] <= fittedAlphap <= expectedRange[[2]];

(* Significance interpretation *)
If[Abs[zScoreToTypical] < 1,
  significance = "Consistent with expectations",
  If[Abs[zScoreToTypical] < 2,
    significance = "Moderately different from expectations",
    significance = "Significantly different from expectations"
  ]
];

Print["Chew-Frautschi Expectations:"];
Print["  Fitted α'' = ", NumberForm[fittedAlphap, {4, 3}], " ± ", NumberForm[fittedAlphapErr, {4, 3}], " GeV⁻²"];
Print["  Expected range: ", expectedRange[[1]], " - ", expectedRange[[2]], " GeV⁻²"];
Print["  Typical value: ", typicalValue, " GeV⁻²"];
Print["  Z-score vs typical: ", NumberForm[zScoreToTypical, {3, 2}]];
Print["  Within expected range: ", If[withinRange, "Yes", "No"]];
Print["  Significance: ", significance];

(* 2. Parity Separation Analysis *)

(* Split data by parity *)
positiveParityData = Select[reggeData, #[[3]] == 1 &];  (* Assuming parity is in column 3 *)
negativeParityData = Select[reggeData, #[[3]] == -1 &];

Print["Parity separation: ", Length[positiveParityData], " positive, ", Length[negativeParityData], " negative parity states"];

(* Fit positive parity trajectory *)
If[Length[positiveParityData] >= 3,
  posFit = LinearModelFit[positiveParityData[[All, {1, 2}]], x, x, 
            Weights -> 1/(positiveParityData[[All, 4]]^2)];  (* Assuming uncertainties in column 4 *)
  posAlphap = posFit["BestFitParameters"][[2]];
  posAlphapErr = posFit["ParameterStandardErrors"][[2]];
  Print["Positive parity: α'' = ", NumberForm[posAlphap, {4, 3}], " ± ", NumberForm[posAlphapErr, {4, 3}], " GeV⁻²"];
];

(* Fit negative parity trajectory *)
If[Length[negativeParityData] >= 3,
  negFit = LinearModelFit[negativeParityData[[All, {1, 2}]], x, x, 
            Weights -> 1/(negativeParityData[[All, 4]]^2)];
  negAlphap = negFit["BestFitParameters"][[2]];
  negAlphapErr = negFit["ParameterStandardErrors"][[2]];
  Print["Negative parity: α'' = ", NumberForm[negAlphap, {4, 3}], " ± ", NumberForm[negAlphapErr, {4, 3}], " GeV⁻²"];
];

(* Compare slopes if both fits are available *)
If[ValueQ[posFit] && ValueQ[negFit],
  slopeDiff = posAlphap - negAlphap;
  slopeDiffErr = Sqrt[posAlphapErr^2 + negAlphapErr^2];
  zScore = slopeDiff / slopeDiffErr;
  
  (* P-value for two-sided test *)
  pValue = 2 (1 - CDF[NormalDistribution[0, 1], Abs[zScore]]);
  
  (* Significance interpretation *)
  If[pValue < 0.001,
    paritySignificance = "Highly significant difference",
    If[pValue < 0.01,
      paritySignificance = "Significant difference",
      If[pValue < 0.05,
        paritySignificance = "Moderately significant difference",
        paritySignificance = "No significant difference"
      ]
    ]
  ];
  
  Print["Parity comparison:"];
  Print["  Δα'' = ", NumberForm[slopeDiff, {4, 3}], " ± ", NumberForm[slopeDiffErr, {4, 3}], " GeV⁻²"];
  Print["  Z-score = ", NumberForm[zScore, {3, 2}]];
  Print["  P-value = ", NumberForm[pValue, {4, 4}]];
  Print["  Significance: ", paritySignificance];
];

(* 3. Radial vs Orbital Trajectory Analysis *)

(* Function to infer radial excitation from particle names *)
inferRadialExcitation[name_String] := Module[
  {radialN},
  If[StringContainsQ[name, "prime"] || StringContainsQ[name, "'"],
    radialN = 1,
    If[StringContainsQ[name, "double prime"] || StringContainsQ[name, "''"],
      radialN = 2,
      (* Look for numbers in name *)
      With[{numbers = StringCases[name, DigitCharacter..]},
        If[Length[numbers] > 0,
          radialN = ToExpression[numbers[[1]]],
          radialN = 0  (* Ground state *)
        ]
      ]
    ]
  ];
  radialN
];

(* Group particles by radial excitation *)
radialGroups = GroupBy[reggeData, inferRadialExcitation[#[[5]]] &];  (* Assuming name is in column 5 *)

Print["Radial excitation analysis:"];
radialFits = <||>;

Do[
  radialN = Keys[radialGroups][[i]];
  radialData = radialGroups[radialN];
  
  If[Length[radialData] >= 3,
    radialFit = LinearModelFit[radialData[[All, {1, 2}]], x, x, 
                Weights -> 1/(radialData[[All, 4]]^2)];
    radialAlphap = radialFit["BestFitParameters"][[2]];
    radialAlphapErr = radialFit["ParameterStandardErrors"][[2]];
    
    radialFits[radialN] = <|
      "alphap" -> radialAlphap,
      "alphap_err" -> radialAlphapErr,
      "n_particles" -> Length[radialData]
    |>;
    
    Print["  n=", radialN, " (n=", Length[radialData], " particles): α'' = ", 
          NumberForm[radialAlphap, {4, 3}], " ± ", NumberForm[radialAlphapErr, {4, 3}], " GeV⁻²"];
  ],
  {i, Length[radialGroups]}
];

(* Test slope universality across radial bands *)
If[Length[radialFits] >= 2,
  (* Extract slopes and uncertainties *)
  slopes = Values[radialFits][[All, "alphap"]];
  slopeErrors = Values[radialFits][[All, "alphap_err"]];
  weights = 1 / (slopeErrors^2);
  
  (* Weighted average *)
  weightedAvg = Total[slopes * weights] / Total[weights];
  weightedAvgErr = Sqrt[1 / Total[weights]];
  
  (* Chi-squared test for universality *)
  chi2 = Total[weights * (slopes - weightedAvg)^2];
  dof = Length[slopes] - 1;
  chi2Dof = chi2 / dof;
  
  (* P-value *)
  pValue = 1 - CDF[ChiSquareDistribution[dof], chi2];
  
  (* Interpretation *)
  If[pValue < 0.05,
    universality = "Rejected - slopes differ significantly",
    universality = "Not rejected - slopes are consistent"
  ];
  
  Print["Slope universality test:"];
  Print["  Weighted average α'' = ", NumberForm[weightedAvg, {4, 3}], " ± ", NumberForm[weightedAvgErr, {4, 3}], " GeV⁻²"];
  Print["  χ²/dof = ", NumberForm[chi2Dof, {3, 2}]];
  Print["  P-value = ", NumberForm[pValue, {4, 4}]];
  Print["  Universality: ", universality];
];

(* 4. Theoretical Implications Summary *)

Print["Theoretical Implications:"];
assessments = {};

(* Chew-Frautschi assessment *)
If[withinRange,
  AppendTo[assessments, "✓ α'' consistent with Chew-Frautschi expectations"],
  AppendTo[assessments, "✗ α'' outside typical range - may indicate new physics or systematic effects"]
];

(* Parity separation assessment *)
If[ValueQ[paritySignificance] && StringContainsQ[paritySignificance, "No significant"],
  AppendTo[assessments, "✓ Parity trajectories consistent - no significant mixing"],
  If[ValueQ[paritySignificance],
    AppendTo[assessments, "✗ Parity trajectories differ - possible mixing or systematics"]
  ]
];

(* Radial universality assessment *)
If[ValueQ[universality] && StringContainsQ[universality, "Not rejected"],
  AppendTo[assessments, "✓ Slope universality across radial bands"],
  If[ValueQ[universality],
    AppendTo[assessments, "✗ Slope universality rejected - radial dependence"]
  ]
];

Do[Print["  ", assessments[[i]]], {i, Length[assessments]}];

(* Export theoretical results for Python comparison *)
theoreticalResults = <|
  "fitted_alphap" -> fittedAlphap,
  "fitted_alphap_err" -> fittedAlphapErr,
  "expected_range" -> expectedRange,
  "typical_value" -> typicalValue,
  "z_score_to_typical" -> zScoreToTypical,
  "within_range" -> withinRange,
  "significance" -> significance,
  "positive_parity_alphap" -> If[ValueQ[posFit], posAlphap, Missing["NotAvailable"]],
  "negative_parity_alphap" -> If[ValueQ[negFit], negAlphap, Missing["NotAvailable"]],
  "parity_slope_difference" -> If[ValueQ[slopeDiff], slopeDiff, Missing["NotAvailable"]],
  "parity_p_value" -> If[ValueQ[pValue], pValue, Missing["NotAvailable"]],
  "radial_fits" -> If[Length[radialFits] > 0, radialFits, Missing["NotAvailable"]],
  "slope_universality" -> If[ValueQ[universality], universality, Missing["NotAvailable"]]
|>;

Export["python_analysis/data_export/theoretical_results.json", theoreticalResults];
Print["Theoretical results exported to theoretical_results.json"]
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_THEORETICAL_OUTPUT = """
Expected theoretical_results.json format:
{
  "fitted_alphap": 0.891,
  "fitted_alphap_err": 0.023,
  "expected_range": [0.8, 1.2],
  "typical_value": 1.0,
  "z_score_to_typical": -0.47,
  "within_range": true,
  "significance": "Consistent with expectations",
  "positive_parity_alphap": 0.894,
  "negative_parity_alphap": 0.887,
  "parity_slope_difference": 0.007,
  "parity_p_value": 0.8234,
  "radial_fits": {
    "0": {"alphap": 0.891, "alphap_err": 0.023, "n_particles": 8},
    "1": {"alphap": 0.893, "alphap_err": 0.031, "n_particles": 4}
  },
  "slope_universality": "Not rejected - slopes are consistent"
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_theoretical_results(json_path: str) -> bool:
    """
    Validate that exported theoretical results meet expected format.
    
    Parameters:
    -----------
    json_path : str
        Path to the exported JSON file
    
    Returns:
    --------
    bool
        True if validation passes
    """
    import json
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    required_keys = [
        'fitted_alphap', 'fitted_alphap_err', 'expected_range', 'typical_value',
        'z_score_to_typical', 'within_range', 'significance'
    ]
    
    missing_keys = set(required_keys) - set(results.keys())
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
        return False
    
    # Check data types
    numeric_keys = [k for k in required_keys if k not in ['within_range', 'significance']]
    for key in numeric_keys:
        if not isinstance(results[key], (int, float)):
            print(f"Key {key} is not numeric: {type(results[key])}")
            return False
    
    print("Theoretical results validation passed!")
    return True

def compare_wl_python_theoretical_results(wl_json_path: str, python_results: dict) -> dict:
    """
    Compare Wolfram Language and Python theoretical results.
    
    Parameters:
    -----------
    wl_json_path : str
        Path to WL theoretical results JSON
    python_results : dict
        Python theoretical analysis results
    
    Returns:
    --------
    dict
        Comparison results
    """
    import json
    
    with open(wl_json_path, 'r') as f:
        wl_results = json.load(f)
    
    comparison = {}
    
    # Compare Chew-Frautschi results
    if 'fitted_alphap' in wl_results and 'cf_results' in python_results:
        wl_alphap = wl_results['fitted_alphap']
        py_alphap = python_results['cf_results']['fitted_alphap']
        diff = abs(wl_alphap - py_alphap)
        rel_diff = diff / wl_alphap if wl_alphap != 0 else float('inf')
        
        comparison['chew_frautschi'] = {
            'wl_alphap': wl_alphap,
            'python_alphap': py_alphap,
            'absolute_difference': diff,
            'relative_difference': rel_diff,
            'agreement': rel_diff < 0.1  # 10% tolerance
        }
    
    # Compare parity separation results
    if 'parity_slope_difference' in wl_results and 'parity_results' in python_results:
        wl_parity_diff = wl_results['parity_slope_difference']
        if wl_parity_diff != 'Missing["NotAvailable"]':
            py_parity_diff = python_results['parity_results']['comparison']['slope_difference']
            diff = abs(wl_parity_diff - py_parity_diff)
            rel_diff = diff / abs(wl_parity_diff) if wl_parity_diff != 0 else float('inf')
            
            comparison['parity_separation'] = {
                'wl_slope_difference': wl_parity_diff,
                'python_slope_difference': py_parity_diff,
                'absolute_difference': diff,
                'relative_difference': rel_diff,
                'agreement': rel_diff < 0.1
            }
    
    return comparison

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE THEORETICAL CONTEXT GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(THEORETICAL_CONTEXT_CODE)
    print("\nExpected output format:")
    print(EXPECTED_THEORETICAL_OUTPUT)
    print("\nThis will provide:")
    print("1. Chew-Frautschi expectations and literature comparison")
    print("2. Parity/naturality separation analysis")
    print("3. Radial vs orbital trajectory analysis")
    print("4. Theoretical implications summary")
