"""
Wolfram Language Statistical Significance Analysis Guide

This module provides code snippets to be added to the Wolfram Language notebook
for implementing statistical significance analysis to ensure robust results.
"""

# =============================================================================
# WOLFRAM LANGUAGE STATISTICAL ANALYSIS CODE (Add to your notebook)
# =============================================================================

STATISTICAL_ANALYSIS_CODE = '''
(* ===== Statistical Significance Analysis for Δ Baryons ===== *)

(* 1. Weighted + Orthogonal Distance Regression Comparison *)

(* Keep your existing weighted linear fit *)
(* weights = 1/σM²² *)
wlsFit = LinearModelFit[reggeData, x, x, Weights -> weights];

(* Add orthogonal distance regression using FindFit with custom loss function *)
(* Define perpendicular distance loss function *)
perpendicularDistance[data_, alpha0_, alphap_] := Module[
  {x, y, predicted, perpendicular},
  x = data[[All, 1]];  (* M² values *)
  y = data[[All, 2]];  (* J values *)
  predicted = alpha0 + alphap * x;
  
  (* Perpendicular distance: |y - predicted| / Sqrt[1 + alphap^2] *)
  perpendicular = Abs[y - predicted] / Sqrt[1 + alphap^2];
  Total[perpendicular]
];

(* Fit using perpendicular distance minimization *)
odrFit = FindMinimum[
  perpendicularDistance[reggeData, alpha0, alphap],
  {{alpha0, wlsFit["BestFitParameters"][[1]]}, 
   {alphap, wlsFit["BestFitParameters"][[2]]}}
];

(* Extract ODR parameters *)
odrAlpha0 = odrFit[[2, 1, 2]];
odrAlphap = odrFit[[2, 2, 2]];

(* Compare results *)
Print["Fitting Method Comparison:"];
Print["WLS:  α₀ = ", NumberForm[wlsFit["BestFitParameters"][[1]], {4, 3}], 
      " ± ", NumberForm[wlsFit["ParameterStandardErrors"][[1]], {4, 3}]];
Print["WLS:  α'' = ", NumberForm[wlsFit["BestFitParameters"][[2]], {4, 3}], 
      " ± ", NumberForm[wlsFit["ParameterStandardErrors"][[2]], {4, 3}]];
Print["ODR:  α₀ = ", NumberForm[odrAlpha0, {4, 3}]];
Print["ODR:  α'' = ", NumberForm[odrAlphap, {4, 3}]];

(* Consistency check *)
alphapDiff = Abs[wlsFit["BestFitParameters"][[2]] - odrAlphap];
alphapCombinedErr = Sqrt[wlsFit["ParameterStandardErrors"][[2]]^2 + 0.01^2]; (* Estimate ODR error *)
consistency = alphapDiff / alphapCombinedErr;
Print["Consistency: Δα'' = ", NumberForm[alphapDiff, {4, 3}], " (", NumberForm[consistency, {3, 1}], "σ)"];

(* 2. Bootstrap & Leave-One-Out Robustness *)

(* Parametric bootstrap: resample with uncertainties *)
nBootstrap = 1000;
bootstrapResults = Table[
  Module[
    {bootData, bootFit, alpha0, alphap},
    (* Resample with noise based on uncertainties *)
    bootData = Table[
      {reggeData[[i, 1]] + RandomVariate[NormalDistribution[0, uncertainties[[i]]]],
       reggeData[[i, 2]] + RandomVariate[NormalDistribution[0, 0.1]]},
      {i, Length[reggeData]}
    ];
    
    (* Fit to bootstrap sample *)
    bootFit = LinearModelFit[bootData, x, x, Weights -> weights];
    alpha0 = bootFit["BestFitParameters"][[1]];
    alphap = bootFit["BestFitParameters"][[2]];
    {alpha0, alphap}
  ],
  {nBootstrap}
];

(* Analyze bootstrap results *)
bootstrapAlpha0 = bootstrapResults[[All, 1]];
bootstrapAlphap = bootstrapResults[[All, 2]];

Print["Bootstrap Analysis:"];
Print["α₀ = ", NumberForm[Mean[bootstrapAlpha0], {4, 3}], " ± ", NumberForm[StandardDeviation[bootstrapAlpha0], {4, 3}]];
Print["α'' = ", NumberForm[Mean[bootstrapAlphap], {4, 3}], " ± ", NumberForm[StandardDeviation[bootstrapAlphap], {4, 3}]];

(* Leave-one-out analysis *)
looResults = Table[
  Module[
    {trainData, testData, trainFit, alpha0, alphap},
    (* Split data *)
    trainData = Delete[reggeData, i];
    testData = reggeData[[i]];
    
    (* Fit on training data *)
    trainFit = LinearModelFit[trainData, x, x, Weights -> Delete[weights, i]];
    alpha0 = trainFit["BestFitParameters"][[1]];
    alphap = trainFit["BestFitParameters"][[2]];
    {alpha0, alphap}
  ],
  {i, Length[reggeData]}
];

(* Analyze LOO results *)
looAlpha0 = looResults[[All, 1]];
looAlphap = looResults[[All, 2]];

Print["Leave-One-Out Analysis:"];
Print["α₀ = ", NumberForm[Mean[looAlpha0], {4, 3}], " ± ", NumberForm[StandardDeviation[looAlpha0], {4, 3}]];
Print["α'' = ", NumberForm[Mean[looAlphap], {4, 3}], " ± ", NumberForm[StandardDeviation[looAlphap], {4, 3}]];

(* Find most influential point *)
looAlphapDeviations = Abs[looAlphap - Mean[looAlphap]];
maxInfluenceIdx = Position[looAlphapDeviations, Max[looAlphapDeviations]][[1, 1]];
Print["Most influential point: ", maxInfluenceIdx];

(* 3. Model Comparison (Linear vs Broken Line) *)

(* Define broken line model *)
brokenLineModel[x_, alpha0_, alphap1_, alphap2_, breakpoint_] := 
  If[x <= breakpoint, 
    alpha0 + alphap1 * x,
    alpha0 + alphap1 * breakpoint + alphap2 * (x - breakpoint)
  ];

(* Fit broken line model *)
brokenLineFit = NonlinearModelFit[
  reggeData,
  brokenLineModel[x, alpha0, alphap1, alphap2, breakpoint],
  {{alpha0, wlsFit["BestFitParameters"][[1]]},
   {alphap1, wlsFit["BestFitParameters"][[2]]},
   {alphap2, wlsFit["BestFitParameters"][[2]] * 0.8},
   {breakpoint, Median[reggeData[[All, 1]]]}},
  x,
  Weights -> weights
];

(* Calculate AIC *)
wlsAIC = 2 * 2 + wlsFit["AIC"];  (* 2 parameters *)
brokenLineAIC = 2 * 4 + brokenLineFit["AIC"];  (* 4 parameters *)
deltaAIC = brokenLineAIC - wlsAIC;

Print["Model Comparison:"];
Print["Linear AIC: ", NumberForm[wlsAIC, {5, 1}]];
Print["Broken Line AIC: ", NumberForm[brokenLineAIC, {5, 1}]];
Print["ΔAIC: ", NumberForm[deltaAIC, {4, 1}]];

(* Model selection interpretation *)
If[deltaAIC < -2,
  modelPreference = "Broken line strongly preferred",
  If[deltaAIC < 0,
    modelPreference = "Broken line weakly preferred",
    If[deltaAIC < 2,
      modelPreference = "No strong preference",
      If[deltaAIC < 7,
        modelPreference = "Linear model weakly preferred",
        modelPreference = "Linear model strongly preferred"
      ]
    ]
  ]
];
Print["Model preference: ", modelPreference];

(* 4. Multiple Testing Control *)

(* If analyzing multiple families, apply FDR correction *)
(* Example: p-values from residual tests *)
pValues = Table[
  (* Calculate p-value for each family/trajectory *)
  (* This is a placeholder - replace with actual p-value calculation *)
  RandomReal[],
  {10}  (* Example: 10 different families *)
];

(* Apply Benjamini-Hochberg FDR correction *)
sortedPValues = Sort[pValues];
nTests = Length[pValues];
criticalValues = Table[i/nTests * 0.05, {i, nTests}];

(* Find significant tests *)
significantTests = Select[
  Transpose[{Range[nTests], sortedPValues, criticalValues}],
  #[[2]] <= #[[3]] &
];

Print["Multiple Testing Control:"];
Print["Original significant tests: ", Count[pValues, _?(# <= 0.05 &)]];
Print["FDR-corrected significant tests: ", Length[significantTests]];

(* 5. Statistical Significance Summary *)

(* Define robustness criteria *)
robustnessChecks = {
  "fitting_methods_consistent" -> (consistency < 2.0),
  "bootstrap_stable" -> (StandardDeviation[bootstrapAlphap] < 0.1),
  "loo_stable" -> (Max[Abs[looAlphap - Mean[looAlphap]]] < 0.05),
  "linear_model_adequate" -> (deltaAIC > -2)
};

nRobust = Count[robustnessChecks[[All, 2]], True];
totalChecks = Length[robustnessChecks];

Print["Statistical Significance Summary:"];
Print["Robustness checks passed: ", nRobust, "/", totalChecks];
Do[
  Print["  ", robustnessChecks[[i, 1]], ": ", If[robustnessChecks[[i, 2]], "✓", "✗"]],
  {i, Length[robustnessChecks]}
];

(* Overall conclusion *)
If[nRobust >= 3,
  conclusion = "Results are statistically robust",
  If[nRobust >= 2,
    conclusion = "Results show moderate robustness",
    conclusion = "Results require caution - limited robustness"
  ]
];
Print["Overall conclusion: ", conclusion];

(* Export results for Python comparison *)
statisticalResults = <|
  "wls_alpha0" -> wlsFit["BestFitParameters"][[1]],
  "wls_alphap" -> wlsFit["BestFitParameters"][[2]],
  "wls_alpha0_err" -> wlsFit["ParameterStandardErrors"][[1]],
  "wls_alphap_err" -> wlsFit["ParameterStandardErrors"][[2]],
  "odr_alpha0" -> odrAlpha0,
  "odr_alphap" -> odrAlphap,
  "bootstrap_alpha0_mean" -> Mean[bootstrapAlpha0],
  "bootstrap_alpha0_std" -> StandardDeviation[bootstrapAlpha0],
  "bootstrap_alphap_mean" -> Mean[bootstrapAlphap],
  "bootstrap_alphap_std" -> StandardDeviation[bootstrapAlphap],
  "loo_alphap_mean" -> Mean[looAlphap],
  "loo_alphap_std" -> StandardDeviation[looAlphap],
  "consistency_check" -> consistency,
  "delta_aic" -> deltaAIC,
  "model_preference" -> modelPreference,
  "robustness_score" -> nRobust,
  "conclusion" -> conclusion
|>;

Export["python_analysis/data_export/statistical_results.json", statisticalResults];
Print["Statistical results exported to statistical_results.json"]
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_STATISTICAL_OUTPUT = """
Expected statistical_results.json format:
{
  "wls_alpha0": 0.523,
  "wls_alphap": 0.891,
  "wls_alpha0_err": 0.045,
  "wls_alphap_err": 0.023,
  "odr_alpha0": 0.518,
  "odr_alphap": 0.894,
  "bootstrap_alpha0_mean": 0.524,
  "bootstrap_alpha0_std": 0.047,
  "bootstrap_alphap_mean": 0.892,
  "bootstrap_alphap_std": 0.025,
  "loo_alphap_mean": 0.893,
  "loo_alphap_std": 0.024,
  "consistency_check": 0.8,
  "delta_aic": 1.2,
  "model_preference": "Linear model weakly preferred",
  "robustness_score": 4,
  "conclusion": "Results are statistically robust"
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_statistical_results(json_path: str) -> bool:
    """
    Validate that exported statistical results meet expected format.
    
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
        'wls_alpha0', 'wls_alphap', 'wls_alpha0_err', 'wls_alphap_err',
        'odr_alpha0', 'odr_alphap',
        'bootstrap_alpha0_mean', 'bootstrap_alpha0_std',
        'bootstrap_alphap_mean', 'bootstrap_alphap_std',
        'loo_alphap_mean', 'loo_alphap_std',
        'consistency_check', 'delta_aic', 'model_preference',
        'robustness_score', 'conclusion'
    ]
    
    missing_keys = set(required_keys) - set(results.keys())
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
        return False
    
    # Check data types
    numeric_keys = [k for k in required_keys if k not in ['model_preference', 'conclusion']]
    for key in numeric_keys:
        if not isinstance(results[key], (int, float)):
            print(f"Key {key} is not numeric: {type(results[key])}")
            return False
    
    print("Statistical results validation passed!")
    return True

def compare_wl_python_statistical_results(wl_json_path: str, python_results: dict) -> dict:
    """
    Compare Wolfram Language and Python statistical results.
    
    Parameters:
    -----------
    wl_json_path : str
        Path to WL statistical results JSON
    python_results : dict
        Python statistical analysis results
    
    Returns:
    --------
    dict
        Comparison results
    """
    import json
    
    with open(wl_json_path, 'r') as f:
        wl_results = json.load(f)
    
    comparison = {}
    
    # Compare key parameters
    for param in ['wls_alphap', 'bootstrap_alphap_mean', 'loo_alphap_mean']:
        if param in wl_results and param in python_results:
            wl_val = wl_results[param]
            py_val = python_results[param]
            diff = abs(wl_val - py_val)
            rel_diff = diff / wl_val if wl_val != 0 else float('inf')
            
            comparison[param] = {
                'wl_value': wl_val,
                'python_value': py_val,
                'absolute_difference': diff,
                'relative_difference': rel_diff,
                'agreement': rel_diff < 0.1  # 10% tolerance
            }
    
    # Compare conclusions
    comparison['conclusions'] = {
        'wl_conclusion': wl_results.get('conclusion', 'Unknown'),
        'python_conclusion': python_results.get('conclusion', 'Unknown'),
        'agreement': wl_results.get('conclusion') == python_results.get('conclusion')
    }
    
    return comparison

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE STATISTICAL ANALYSIS GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(STATISTICAL_ANALYSIS_CODE)
    print("\nExpected output format:")
    print(EXPECTED_STATISTICAL_OUTPUT)
    print("\nThis will provide:")
    print("1. Weighted vs ODR fitting comparison")
    print("2. Bootstrap and LOO robustness tests")
    print("3. Linear vs broken line model comparison")
    print("4. Multiple testing control")
    print("5. Statistical significance summary")
