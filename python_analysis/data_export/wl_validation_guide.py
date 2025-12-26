"""
Wolfram Language Validation Analysis Guide

This module provides code snippets to be added to the Wolfram Language notebook
for implementing validation analysis to compare findings with experimental data and theoretical expectations.
"""

# =============================================================================
# WOLFRAM LANGUAGE VALIDATION CODE (Add to your notebook)
# =============================================================================

VALIDATION_ANALYSIS_CODE = '''
(* ===== Validation Analysis ===== *)

(* 1. PDG Cross-check Near Predictions *)

(* Function to find nearby PDG candidates *)
findNearbyCandidates[pred_, windowFactor_: 2.0] := Module[
  {Jpred, Mpred, Msigma, window, Mmin, Mmax, candidates, distances},
  Jpred = pred[[1]];  (* J value *)
  Mpred = pred[[2]];  (* predicted mass *)
  Msigma = pred[[3]]; (* mass uncertainty *)
  
  window = windowFactor * Msigma;
  Mmin = Mpred - window;
  Mmax = Mpred + window;
  
  (* Find candidates with same J and mass in window *)
  candidates = Select[reggeData,
    #[[2]] == Jpred &&  (* same J *)
    Sqrt[#[[1]]] >= Mmin && Sqrt[#[[1]]] <= Mmax &  (* mass in window *)
  ];
  
  If[Length[candidates] > 0,
    (* Calculate distances and sort *)
    distances = Map[{#, Abs[Sqrt[#[[1]]] - Mpred]}&, candidates];
    distances = SortBy[distances, #[[2]]&];
    
    (* Categorize match quality *)
    bestDistance = distances[[1, 2]];
    If[bestDistance <= Msigma,
      matchType = "confirmation",
      If[bestDistance <= 2 * Msigma,
        matchType = "near-miss",
        matchType = "distant"
      ]
    ];
    
    {candidates, distances, matchType},
    (* No candidates found *)
    {{}, {}, "genuine_gap"}
  ]
];

(* Cross-check all predictions *)
crosscheckResults = Table[
  Module[{result},
    result = findNearbyCandidates[pred];
    <|
      "J_predicted" -> pred[[1]],
      "M_predicted" -> pred[[2]],
      "M_sigma" -> pred[[3]],
      "candidates" -> result[[1]],
      "distances" -> result[[2]],
      "match_type" -> result[[3]]
    |>
  ],
  {pred, PredictedStates}
];

(* Summary statistics *)
nPredictions = Length[crosscheckResults];
nConfirmations = Count[crosscheckResults, _?(#["match_type"] == "confirmation"&)];
nNearMisses = Count[crosscheckResults, _?(#["match_type"] == "near-miss"&)];
nGaps = Count[crosscheckResults, _?(#["match_type"] == "genuine_gap"&)];

Print["PDG Cross-check Results:"];
Print["  Total predictions: ", nPredictions];
Print["  Confirmations: ", nConfirmations];
Print["  Near-misses: ", nNearMisses];
Print["  Genuine gaps: ", nGaps];

(* Detailed results table *)
Print["Detailed cross-check results:"];
Do[
  result = crosscheckResults[[i]];
  Print["  J = ", result["J_predicted"], ":"];
  Print["    Predicted: ", NumberForm[result["M_predicted"], {4, 3}], " ± ", 
        NumberForm[result["M_sigma"], {4, 3}], " GeV"];
  
  If[result["match_type"] != "genuine_gap",
    bestCandidate = result["candidates"][[1]];
    bestDistance = result["distances"][[1, 2]];
    Print["    Best match: ", bestCandidate[[5]], " (", NumberForm[Sqrt[bestCandidate[[1]]], {4, 3}], " GeV)"];
    Print["    Distance: ", NumberForm[bestDistance, {4, 3}], " GeV"];
    Print["    Type: ", result["match_type"]],
    Print["    No nearby PDG candidates found"]
  ];
  Print[""],
  {i, Length[crosscheckResults]}
];

(* 2. Residuals vs Experimental Quality *)

(* Calculate residuals *)
residuals = y - lm[x];
absResiduals = Abs[residuals];

(* Extract quality indicators *)
qualityData = Table[
  Module[{width, status, massUncertainty, totalUncertainty},
    width = If[ValueQ[reggeData[[i, 6]]], reggeData[[i, 6]], Missing["NotAvailable"]];  (* Assuming width in column 6 *)
    status = If[ValueQ[reggeData[[i, 7]]], reggeData[[i, 7]], "Unknown"];  (* Assuming status in column 7 *)
    massUncertainty = reggeData[[i, 4]];  (* M² uncertainty *)
    widthUncertainty = If[NumberQ[width], 0.25 * width, 0.0];
    totalUncertainty = Sqrt[massUncertainty^2 + widthUncertainty^2];
    
    <|
      "residual" -> residuals[[i]],
      "abs_residual" -> absResiduals[[i]],
      "width" -> width,
      "status" -> status,
      "total_uncertainty" -> totalUncertainty
    |>
  ],
  {i, Length[reggeData]}
];

(* Correlation analysis *)
validData = Select[qualityData, NumberQ[#["total_uncertainty"]] && #["total_uncertainty"] > 0 &];

If[Length[validData] >= 3,
  (* Residual vs width correlation *)
  widths = validData[[All, "width"]];
  validWidths = Select[widths, NumberQ[#] && # > 0 &];
  If[Length[validWidths] > 0,
    widthCorr = Correlation[validWidths, Select[absResiduals, NumberQ[#] &][[;; Length[validWidths]]]];
    Print["Residual vs width correlation: r = ", NumberForm[widthCorr, {4, 3}]];
  ];
  
  (* Residual vs uncertainty correlation *)
  uncertainties = validData[[All, "total_uncertainty"]];
  uncertaintyCorr = Correlation[uncertainties, validData[[All, "abs_residual"]]];
  Print["Residual vs uncertainty correlation: r = ", NumberForm[uncertaintyCorr, {4, 3}]];
  
  (* ANOVA by status groups *)
  statusGroups = GroupBy[validData, #["status"]&];
  If[Length[statusGroups] >= 2,
    groupResiduals = Values[statusGroups][[All, All, "abs_residual"]];
    fStat = ANOVA[groupResiduals][[1, 4]];
    pValue = ANOVA[groupResiduals][[1, 5]];
    Print["ANOVA by status groups: F = ", NumberForm[fStat, {4, 3}], ", p = ", NumberForm[pValue, {4, 4}]];
  ];
];

(* 3. External Theory Overlay *)

(* Canonical theoretical parameters *)
canonicalAlpha0 = 0.0;  (* Adjust based on literature *)
canonicalAlphap = 0.9;  (* GeV⁻², typical baryon value *)

(* Calculate canonical predictions *)
yCanonical = canonicalAlpha0 + canonicalAlphap * x;

(* Calculate deviations *)
deviations = y - yCanonical;
absDeviations = Abs[deviations];

(* RMS deviation *)
rmsDeviation = Sqrt[Mean[deviations^2]];

(* Weighted RMS *)
weights = 1 / (σx^2);
weightedRMS = Sqrt[Total[weights * deviations^2] / Total[weights]];

(* Statistical significance *)
meanDeviation = Mean[deviations];
stdDeviation = StandardDeviation[deviations];
zScore = meanDeviation / (stdDeviation / Sqrt[Length[deviations]]);

(* Agreement assessment *)
If[Abs[zScore] < 1,
  agreement = "Excellent agreement",
  If[Abs[zScore] < 2,
    agreement = "Good agreement",
    If[Abs[zScore] < 3,
      agreement = "Moderate agreement",
      agreement = "Poor agreement - significant tension"
    ]
  ]
];

Print["External Theory Comparison:"];
Print["  Fitted trajectory: J = ", NumberForm[α0, {4, 3}], " + ", NumberForm[αp, {4, 3}], " M²"];
Print["  Canonical theory: J = ", NumberForm[canonicalAlpha0, {4, 3}], " + ", NumberForm[canonicalAlphap, {4, 3}], " M²"];
Print["  RMS deviation: ", NumberForm[rmsDeviation, {4, 4}]];
Print["  Weighted RMS: ", NumberForm[weightedRMS, {4, 4}]];
Print["  Z-score: ", NumberForm[zScore, {3, 2}]];
Print["  Agreement: ", agreement];

(* 4. Validation Summary *)

Print["Validation Assessment:"];

(* PDG cross-check assessment *)
confirmationRate = nConfirmations / nPredictions;
If[confirmationRate >= 0.5,
  Print["  ✓ Good PDG cross-check confirmation rate"],
  If[confirmationRate >= 0.2,
    Print["  ⚠ Moderate PDG cross-check confirmation rate"],
    Print["  ✗ Low PDG cross-check confirmation rate"]
  ]
];

(* Residual quality assessment *)
If[ValueQ[uncertaintyCorr] && Abs[uncertaintyCorr] > 0.3,
  Print["  ✓ Residuals correlate with experimental quality - measurement effects identified"],
  Print["  ✗ Residuals independent of experimental quality - possible systematic effects"]
];

(* Theory agreement assessment *)
If[Abs[zScore] < 2,
  Print["  ✓ Good agreement with canonical theory"],
  If[Abs[zScore] < 3,
    Print["  ⚠ Moderate agreement with canonical theory"],
    Print["  ✗ Poor agreement with canonical theory - significant tension"]
  ]
];

(* Export validation results for Python comparison *)
validationResults = <|
  "pdg_crosscheck" -> <|
    "n_predictions" -> nPredictions,
    "n_confirmations" -> nConfirmations,
    "n_near_misses" -> nNearMisses,
    "n_gaps" -> nGaps,
    "confirmation_rate" -> confirmationRate
  |>,
  "residual_quality" -> <|
    "width_correlation" -> If[ValueQ[widthCorr], widthCorr, Missing["NotAvailable"]],
    "uncertainty_correlation" -> If[ValueQ[uncertaintyCorr], uncertaintyCorr, Missing["NotAvailable"]],
    "anova_f_stat" -> If[ValueQ[fStat], fStat, Missing["NotAvailable"]],
    "anova_p_value" -> If[ValueQ[pValue], pValue, Missing["NotAvailable"]]
  |>,
  "theory_comparison" -> <|
    "rms_deviation" -> rmsDeviation,
    "weighted_rms" -> weightedRMS,
    "z_score" -> zScore,
    "agreement" -> agreement
  |>
|>;

Export["python_analysis/data_export/validation_results.json", validationResults];
Print["Validation results exported to validation_results.json"]
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_VALIDATION_OUTPUT = """
Expected validation_results.json format:
{
  "pdg_crosscheck": {
    "n_predictions": 8,
    "n_confirmations": 3,
    "n_near_misses": 2,
    "n_gaps": 3,
    "confirmation_rate": 0.375
  },
  "residual_quality": {
    "width_correlation": 0.234,
    "uncertainty_correlation": 0.456,
    "anova_f_stat": 2.34,
    "anova_p_value": 0.1234
  },
  "theory_comparison": {
    "rms_deviation": 0.1234,
    "weighted_rms": 0.1123,
    "z_score": 1.23,
    "agreement": "Good agreement"
  }
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_validation_results(json_path: str) -> bool:
    """
    Validate that exported validation results meet expected format.
    
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
    
    required_sections = ['pdg_crosscheck', 'residual_quality', 'theory_comparison']
    
    for section in required_sections:
        if section not in results:
            print(f"Missing section: {section}")
            return False
    
    # Check PDG cross-check section
    pdg_keys = ['n_predictions', 'n_confirmations', 'n_near_misses', 'n_gaps', 'confirmation_rate']
    for key in pdg_keys:
        if key not in results['pdg_crosscheck']:
            print(f"Missing PDG cross-check key: {key}")
            return False
    
    # Check theory comparison section
    theory_keys = ['rms_deviation', 'weighted_rms', 'z_score', 'agreement']
    for key in theory_keys:
        if key not in results['theory_comparison']:
            print(f"Missing theory comparison key: {key}")
            return False
    
    print("Validation results validation passed!")
    return True

def compare_wl_python_validation_results(wl_json_path: str, python_results: dict) -> dict:
    """
    Compare Wolfram Language and Python validation results.
    
    Parameters:
    -----------
    wl_json_path : str
        Path to WL validation results JSON
    python_results : dict
        Python validation analysis results
    
    Returns:
    --------
    dict
        Comparison results
    """
    import json
    
    with open(wl_json_path, 'r') as f:
        wl_results = json.load(f)
    
    comparison = {}
    
    # Compare PDG cross-check results
    if 'pdg_crosscheck' in wl_results and 'validation_results' in python_results:
        wl_pdg = wl_results['pdg_crosscheck']
        py_pdg = python_results['validation_results']['crosscheck_results']
        
        wl_confirmations = wl_pdg['n_confirmations']
        py_confirmations = len(py_pdg[py_pdg['match_type'] == 'confirmation'])
        
        diff = abs(wl_confirmations - py_confirmations)
        agreement = diff == 0
        
        comparison['pdg_crosscheck'] = {
            'wl_confirmations': wl_confirmations,
            'python_confirmations': py_confirmations,
            'difference': diff,
            'agreement': agreement
        }
    
    # Compare theory comparison results
    if 'theory_comparison' in wl_results and 'validation_results' in python_results:
        wl_theory = wl_results['theory_comparison']
        py_theory = python_results['validation_results']['theory_results']
        
        wl_rms = wl_theory['rms_deviation']
        py_rms = py_theory['rms_deviation']
        
        rel_diff = abs(wl_rms - py_rms) / wl_rms if wl_rms != 0 else float('inf')
        agreement = rel_diff < 0.1  # 10% tolerance
        
        comparison['theory_comparison'] = {
            'wl_rms': wl_rms,
            'python_rms': py_rms,
            'relative_difference': rel_diff,
            'agreement': agreement
        }
    
    return comparison

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE VALIDATION ANALYSIS GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(VALIDATION_ANALYSIS_CODE)
    print("\nExpected output format:")
    print(EXPECTED_VALIDATION_OUTPUT)
    print("\nThis will provide:")
    print("1. PDG cross-check near predictions")
    print("2. Residuals vs experimental quality correlation")
    print("3. External theory overlay comparison")
    print("4. Overall validation assessment")
