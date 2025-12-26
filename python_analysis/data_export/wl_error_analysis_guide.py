"""
Wolfram Language Error Analysis Guide

This module provides code snippets to be added to the Wolfram Language notebook
for implementing systematic uncertainty analysis including width-based systematics,
uncertainty propagation, and sensitivity analysis for Regge trajectories.
"""

# =============================================================================
# WOLFRAM LANGUAGE ERROR ANALYSIS CODE (Add to your notebook)
# =============================================================================

ERROR_ANALYSIS_CODE = '''
(* ===== Systematic Uncertainty Analysis ===== *)

(* 1. Systematic from width *)

(* Configuration for systematic uncertainty analysis *)
errorAnalysisConfig = <|
  "kappaRange" -> {0.0, 0.5},
  "kappaSteps" -> 21,
  "defaultKappa" -> 0.25,
  "confidenceLevel" -> 0.68,  (* 1σ *)
  "minWidthThreshold" -> 0.001,  (* GeV *)
  "maxWidthFraction" -> 0.5
|>;

(* Compute systematic uncertainties from resonance widths *)
computeSystematicUncertainties[data_, kappa_: 0.25] := Module[
  {resultData, systematicErrors, totalUncertainties, massSquaredUncertainties},
  
  resultData = data;
  systematicErrors = {};
  totalUncertainties = {};
  massSquaredUncertainties = {};
  
  Do[
    With[{mass = row["MassGeV"], 
          massUncertainty = Lookup[row, "MassSigmaGeV", 0.0],
          width = Lookup[row, "ResonanceWidthGeV", 0.0]},
      
      (* Compute width-based systematic *)
      systematicError = If[width > errorAnalysisConfig["minWidthThreshold"],
        (* Limit width contribution to reasonable fraction of mass *)
        maxWidthContribution = mass * errorAnalysisConfig["maxWidthFraction"];
        widthContribution = Min[width * kappa, maxWidthContribution];
        widthContribution,
        0.0
      ];
      
      (* Combine uncertainties in quadrature *)
      totalUncertainty = Sqrt[massUncertainty^2 + systematicError^2];
      
      (* Propagate to mass-squared uncertainty *)
      massSquaredUncertainty = 2 * mass * totalUncertainty;
      
      AppendTo[systematicErrors, systematicError];
      AppendTo[totalUncertainties, totalUncertainty];
      AppendTo[massSquaredUncertainties, massSquaredUncertainty];
    ],
    {row, resultData}
  ];
  
  (* Add uncertainty columns *)
  resultData = MapThread[
    Append[#1, <|
      "SystematicErrorGeV" -> #2,
      "TotalUncertaintyGeV" -> #3,
      "M2SigmaGeV2" -> #4,
      "Kappa" -> kappa
    |>] &,
    {resultData, systematicErrors, totalUncertainties, massSquaredUncertainties}
  ];
  
  resultData
];

(* 2. Sensitivity analysis of α' vs κ *)

(* Analyze sensitivity of fitted parameters to κ *)
analyzeKappaSensitivity[data_, fitFunction_, fitParams_] := Module[
  {kappaValues, sensitivityResults, paramStorage},
  
  kappaValues = Range[errorAnalysisConfig["kappaRange"][[1]], 
                     errorAnalysisConfig["kappaRange"][[2]], 
                     (errorAnalysisConfig["kappaRange"][[2]] - errorAnalysisConfig["kappaRange"][[1]]) / 
                     (errorAnalysisConfig["kappaSteps"] - 1)];
  
  sensitivityResults = <|
    "kappaValues" -> kappaValues,
    "fitParameters" -> <||>,
    "fitUncertainties" -> <||>,
    "chi2Values" -> {},
    "rSquaredValues" -> {}
  |>;
  
  (* Initialize parameter storage *)
  Do[
    sensitivityResults["fitParameters"][param] = {};
    sensitivityResults["fitUncertainties"][param] = {},
    {param, fitParams}
  ];
  
  (* Analyze sensitivity across kappa range *)
  Do[
    (* Compute systematic uncertainties for this kappa *)
    dataWithUncertainties = computeSystematicUncertainties[data, kappa];
    
    (* Prepare data for fitting *)
    xData = dataWithUncertainties[[All, "M2GeV2"]];
    yData = dataWithUncertainties[[All, "J"]];
    yErrors = dataWithUncertainties[[All, "M2SigmaGeV2"]];
    
    (* Remove points with invalid uncertainties *)
    validMask = (yErrors > 0) && (yErrors < Infinity);
    xValid = Pick[xData, validMask];
    yValid = Pick[yData, validMask];
    yErrorsValid = Pick[yErrors, validMask];
    
    If[Length[xValid] >= 2,
      Try[
        (* Perform weighted fit *)
        fitResult = NonlinearModelFit[Transpose[{xValid, yValid}], 
                                     fitFunction[x, ##] & @@ fitParams, 
                                     fitParams, x, 
                                     Weights -> 1 / yErrorsValid^2];
        
        (* Extract parameters and uncertainties *)
        popt = fitResult["BestFitParameters"];
        pcov = fitResult["ParameterCovarianceMatrix"];
        
        (* Compute fit statistics *)
        yPred = fitResult["PredictedResponse"];
        residuals = yValid - yPred;
        chi2 = Total[(residuals / yErrorsValid)^2];
        dof = Length[xValid] - Length[fitParams];
        chi2Dof = If[dof > 0, chi2 / dof, Infinity];
        
        (* Compute R-squared *)
        ssRes = Total[residuals^2];
        ssTot = Total[(yValid - Mean[yValid])^2];
        rSquared = If[ssTot > 0, 1 - ssRes / ssTot, 0];
        
        (* Store results *)
        Do[
          AppendTo[sensitivityResults["fitParameters"][param], popt[[i]]];
          AppendTo[sensitivityResults["fitUncertainties"][param], Sqrt[pcov[[i, i]]]],
          {i, Length[fitParams]},
          {param, fitParams}
        ];
        
        AppendTo[sensitivityResults["chi2Values"], chi2Dof];
        AppendTo[sensitivityResults["rSquaredValues"], rSquared],
        
        (* Handle fit failures *)
        Do[
          AppendTo[sensitivityResults["fitParameters"][param], Indeterminate];
          AppendTo[sensitivityResults["fitUncertainties"][param], Indeterminate],
          {param, fitParams}
        ];
        AppendTo[sensitivityResults["chi2Values"], Indeterminate];
        AppendTo[sensitivityResults["rSquaredValues"], Indeterminate]
      ]
    ],
    {kappa, kappaValues}
  ];
  
  sensitivityResults
];

(* 3. Propagation to predictions *)

(* Propagate fit uncertainties to mass predictions *)
propagateUncertaintiesToPredictions[fitParameters_, fitCovariance_, jValues_, 
                                   fitFunction_, confidenceLevel_: 0.68] := Module[
  {nSigma, predictions, alpha0, alphap, massSquaredPred, massPred, massUncertainty},
  
  nSigma = InverseCDF[NormalDistribution[], (1 + confidenceLevel) / 2];
  predictions = {};
  
  Do[
    Try[
      (* For linear fit: J = α₀ + α'M² → M² = (J - α₀) / α' *)
      {alpha0, alphap} = fitParameters;
      massSquaredPred = (j - alpha0) / alphap;
      
      (* Compute uncertainty using error propagation *)
      (* ∂M²/∂α₀ = -1/α', ∂M²/∂α' = -(J-α₀)/α'² *)
      dDalpha0 = -1 / alphap;
      dDalphap = -(j - alpha0) / (alphap^2);
      
      (* Variance of M² prediction *)
      varMassSquared = dDalpha0^2 * fitCovariance[[1, 1]] + 
                      dDalphap^2 * fitCovariance[[2, 2]] + 
                      2 * dDalpha0 * dDalphap * fitCovariance[[1, 2]];
      
      massSquaredUncertainty = Sqrt[varMassSquared];
      
      (* Convert to mass and mass uncertainty *)
      massPred = Sqrt[massSquaredPred];
      massUncertainty = massSquaredUncertainty / (2 * massPred);
      
      (* Compute prediction intervals *)
      massLower = massPred - nSigma * massUncertainty;
      massUpper = massPred + nSigma * massUncertainty;
      
      AppendTo[predictions, <|
        "J" -> j,
        "PredictedMassGeV" -> massPred,
        "MassUncertaintyGeV" -> massUncertainty,
        "MassLowerGeV" -> massLower,
        "MassUpperGeV" -> massUpper,
        "PredictedM2GeV2" -> massSquaredPred,
        "M2UncertaintyGeV2" -> massSquaredUncertainty,
        "ConfidenceLevel" -> confidenceLevel,
        "NSigma" -> nSigma
      |>],
      
      (* Handle cases where prediction fails *)
      AppendTo[predictions, <|
        "J" -> j,
        "PredictedMassGeV" -> Indeterminate,
        "MassUncertaintyGeV" -> Indeterminate,
        "MassLowerGeV" -> Indeterminate,
        "MassUpperGeV" -> Indeterminate,
        "PredictedM2GeV2" -> Indeterminate,
        "M2UncertaintyGeV2" -> Indeterminate,
        "ConfidenceLevel" -> confidenceLevel,
        "NSigma" -> nSigma
      |>]
    ],
    {j, jValues}
  ];
  
  Dataset[predictions]
];

(* 4. Cross-check predictions with PDG *)

(* Cross-check predictions with PDG data and flag gaps *)
crossCheckPredictionsWithPDG[predictions_, pdgData_, nSigmaThreshold_: 2.0] := Module[
  {crossCheckResults, jValue, predMass, predUncertainty, searchLower, searchUpper,
   nearbyEntries, closestDistance, isSignificantGap, gapAnalysis},
  
  crossCheckResults = {};
  
  Do[
    jValue = prediction["J"];
    predMass = prediction["PredictedMassGeV"];
    predUncertainty = prediction["MassUncertaintyGeV"];
    
    If[!NumberQ[predMass] || !NumberQ[predUncertainty],
      AppendTo[crossCheckResults, <|
        "J" -> jValue,
        "PredictedMassGeV" -> predMass,
        "MassUncertaintyGeV" -> predUncertainty,
        "SearchWindowLower" -> Indeterminate,
        "SearchWindowUpper" -> Indeterminate,
        "NearbyPDGEntries" -> {},
        "ClosestPDGEntry" -> None,
        "ClosestDistanceSigma" -> Indeterminate,
        "IsSignificantGap" -> False,
        "GapAnalysis" -> "Prediction failed"
      |>],
      
      (* Define search window *)
      searchLower = predMass - nSigmaThreshold * predUncertainty;
      searchUpper = predMass + nSigmaThreshold * predUncertainty;
      
      (* Find nearby PDG entries with same J *)
      nearbyEntries = {};
      Do[
        If[pdgRow["J"] == jValue,
          pdgMass = pdgRow["MassGeV"];
          If[NumberQ[pdgMass] && searchLower <= pdgMass <= searchUpper,
            distanceSigma = Abs[pdgMass - predMass] / predUncertainty;
            AppendTo[nearbyEntries, <|
              "PDGEntry" -> pdgRow["Name"],
              "PDGMassGeV" -> pdgMass,
              "DistanceSigma" -> distanceSigma,
              "Status" -> pdgRow["Status"]
            |>]
          ]
        ],
        {pdgRow, pdgData}
      ];
      
      (* Sort by distance *)
      nearbyEntries = SortBy[nearbyEntries, #["DistanceSigma"] &];
      
      (* Determine if this is a significant gap *)
      closestDistance = If[Length[nearbyEntries] > 0, 
                          nearbyEntries[[1, "DistanceSigma"]], 
                          Infinity];
      isSignificantGap = Length[nearbyEntries] == 0 || closestDistance > nSigmaThreshold;
      
      (* Analyze gap *)
      gapAnalysis = If[Length[nearbyEntries] == 0,
        "No PDG entries found within " <> ToString[nSigmaThreshold] <> "σ window",
        If[closestDistance > nSigmaThreshold,
          "Closest PDG entry is " <> ToString[NumberForm[closestDistance, {3, 2}]] <> "σ away",
          "PDG entry found at " <> ToString[NumberForm[closestDistance, {3, 2}]] <> "σ distance"
        ]
      ];
      
      AppendTo[crossCheckResults, <|
        "J" -> jValue,
        "PredictedMassGeV" -> predMass,
        "MassUncertaintyGeV" -> predUncertainty,
        "SearchWindowLower" -> searchLower,
        "SearchWindowUpper" -> searchUpper,
        "NearbyPDGEntries" -> nearbyEntries,
        "ClosestPDGEntry" -> If[Length[nearbyEntries] > 0, nearbyEntries[[1]], None],
        "ClosestDistanceSigma" -> closestDistance,
        "IsSignificantGap" -> isSignificantGap,
        "GapAnalysis" -> gapAnalysis
      |>]
    ],
    {prediction, predictions}
  ];
  
  Dataset[crossCheckResults]
];

(* 5. Create sensitivity plots *)

(* Create sensitivity analysis plots *)
createSensitivityPlots[sensitivityResults_, outputDir_: "error_analysis"] := Module[
  {kappaValues, alphapValues, alphapErrors, alpha0Values, alpha0Errors,
   chi2Values, r2Values, defaultIdx},
  
  kappaValues = sensitivityResults["kappaValues"];
  alphapValues = sensitivityResults["fitParameters"]["alphap"];
  alphapErrors = sensitivityResults["fitUncertainties"]["alphap"];
  alpha0Values = sensitivityResults["fitParameters"]["alpha0"];
  alpha0Errors = sensitivityResults["fitUncertainties"]["alpha0"];
  chi2Values = sensitivityResults["chi2Values"];
  r2Values = sensitivityResults["rSquaredValues"];
  
  (* Find default kappa index *)
  defaultIdx = First[Position[kappaValues, errorAnalysisConfig["defaultKappa"]]];
  If[Length[defaultIdx] == 0,
    defaultIdx = First[Position[kappaValues, Nearest[kappaValues, errorAnalysisConfig["defaultKappa"]]]]
  ];
  
  (* Create multi-panel plot *)
  GraphicsGrid[{
    {
      (* α' vs κ *)
      If[Length[alphapValues] > 0,
        ErrorListPlot[Transpose[{kappaValues, alphapValues, alphapErrors}],
          PlotMarkers -> Automatic, Frame -> True,
          FrameLabel -> {"κ (Width Systematic Fraction)", "α' (GeV⁻²)"},
          PlotLabel -> "Regge Slope vs Width Systematic",
          GridLines -> Automatic, GridLinesStyle -> Directive[Gray, Dashed],
          Epilog -> {Red, Dashed, 
                    Line[{{errorAnalysisConfig["defaultKappa"], Min[alphapValues]}, 
                          {errorAnalysisConfig["defaultKappa"], Max[alphapValues]}}],
                    Text["Default κ", {errorAnalysisConfig["defaultKappa"], Max[alphapValues]}, {-1, 1}]}],
        Graphics[Text["No valid data", {0.5, 0.5}]]
      ],
      
      (* α₀ vs κ *)
      If[Length[alpha0Values] > 0,
        ErrorListPlot[Transpose[{kappaValues, alpha0Values, alpha0Errors}],
          PlotMarkers -> Automatic, Frame -> True,
          FrameLabel -> {"κ (Width Systematic Fraction)", "α₀"},
          PlotLabel -> "Regge Intercept vs Width Systematic",
          GridLines -> Automatic, GridLinesStyle -> Directive[Gray, Dashed],
          Epilog -> {Red, Dashed, 
                    Line[{{errorAnalysisConfig["defaultKappa"], Min[alpha0Values]}, 
                          {errorAnalysisConfig["defaultKappa"], Max[alpha0Values]}}]}],
        Graphics[Text["No valid data", {0.5, 0.5}]]
      ]
    },
    {
      (* χ²/dof vs κ *)
      If[Length[chi2Values] > 0,
        ListPlot[Transpose[{kappaValues, chi2Values}],
          PlotMarkers -> Automatic, Frame -> True,
          FrameLabel -> {"κ (Width Systematic Fraction)", "χ²/dof"},
          PlotLabel -> "Fit Quality vs Width Systematic",
          GridLines -> Automatic, GridLinesStyle -> Directive[Gray, Dashed],
          Epilog -> {Red, Dashed, 
                    Line[{{errorAnalysisConfig["defaultKappa"], Min[chi2Values]}, 
                          {errorAnalysisConfig["defaultKappa"], Max[chi2Values]}}],
                    Green, Dotted, Line[{{Min[kappaValues], 1}, {Max[kappaValues], 1}}],
                    Text["χ²/dof = 1", {Max[kappaValues], 1}, {-1, 1}]}],
        Graphics[Text["No valid data", {0.5, 0.5}]]
      ],
      
      (* R² vs κ *)
      If[Length[r2Values] > 0,
        ListPlot[Transpose[{kappaValues, r2Values}],
          PlotMarkers -> Automatic, Frame -> True,
          FrameLabel -> {"κ (Width Systematic Fraction)", "R²"},
          PlotLabel -> "Goodness of Fit vs Width Systematic",
          GridLines -> Automatic, GridLinesStyle -> Directive[Gray, Dashed],
          Epilog -> {Red, Dashed, 
                    Line[{{errorAnalysisConfig["defaultKappa"], Min[r2Values]}, 
                          {errorAnalysisConfig["defaultKappa"], Max[r2Values]}}]}],
        Graphics[Text["No valid data", {0.5, 0.5}]]
      ]
    }
  }, ImageSize -> 800]
];

(* 6. Generate error analysis report *)

(* Generate comprehensive error analysis report *)
generateErrorAnalysisReport[sensitivityResults_, predictions_, crossCheckResults_, 
                          outputFile_: "error_analysis_report.txt"] := Module[
  {report, kappaValues, alphapValues, chi2Values, optimalIdx, optimalKappa, optimalAlphap,
   defaultIdx, defaultAlphap, defaultChi2, significantGaps},
  
  report = "";
  report = report <> "=" <> StringRepeat["-", 78] <> "\n";
  report = report <> "SYSTEMATIC UNCERTAINTY ANALYSIS REPORT\n";
  report = report <> "=" <> StringRepeat["-", 78] <> "\n\n";
  report = report <> "Analysis Date: " <> DateString[] <> "\n";
  report = report <> "Default κ: " <> ToString[errorAnalysisConfig["defaultKappa"]] <> "\n";
  report = report <> "Confidence Level: " <> ToString[NumberForm[errorAnalysisConfig["confidenceLevel"], {3, 1}]] <> "%\n\n";
  
  (* Kappa sensitivity analysis *)
  report = report <> "1. KAPPA SENSITIVITY ANALYSIS\n";
  report = report <> StringRepeat["-", 50] <> "\n";
  
  kappaValues = sensitivityResults["kappaValues"];
  alphapValues = sensitivityResults["fitParameters"]["alphap"];
  chi2Values = sensitivityResults["chi2Values"];
  
  If[Length[alphapValues] > 0,
    (* Find optimal kappa (minimum chi2) *)
    If[Length[chi2Values] > 0,
      optimalIdx = First[Position[chi2Values, Min[DeleteCases[chi2Values, Indeterminate]]]];
      If[Length[optimalIdx] > 0,
        optimalKappa = kappaValues[[optimalIdx[[1]]]];
        optimalAlphap = alphapValues[[optimalIdx[[1]]]];
        report = report <> "Optimal κ (minimum χ²/dof): " <> ToString[NumberForm[optimalKappa, {3, 3}]] <> "\n";
        report = report <> "Optimal α': " <> ToString[NumberForm[optimalAlphap, {4, 4}]] <> " GeV⁻²\n";
        report = report <> "Minimum χ²/dof: " <> ToString[NumberForm[chi2Values[[optimalIdx[[1]]]], {3, 3}]] <> "\n\n"
      ]
    ];
    
    (* Default kappa analysis *)
    defaultIdx = First[Position[kappaValues, errorAnalysisConfig["defaultKappa"]]];
    If[Length[defaultIdx] > 0,
      defaultAlphap = alphapValues[[defaultIdx[[1]]]];
      defaultChi2 = chi2Values[[defaultIdx[[1]]]];
      report = report <> "Default κ = " <> ToString[errorAnalysisConfig["defaultKappa"]] <> ":\n";
      report = report <> "  α' = " <> ToString[NumberForm[defaultAlphap, {4, 4}]] <> " GeV⁻²\n";
      report = report <> "  χ²/dof = " <> ToString[NumberForm[defaultChi2, {3, 3}]] <> "\n";
      report = report <> "  R² = " <> ToString[NumberForm[sensitivityResults["rSquaredValues"][[defaultIdx[[1]]]], {4, 4}]] <> "\n\n"
    ];
    
    (* Parameter range analysis *)
    validAlphap = DeleteCases[alphapValues, Indeterminate];
    If[Length[validAlphap] > 0,
      report = report <> "α' range across κ: [" <> ToString[NumberForm[Min[validAlphap], {4, 4}]] <> 
                ", " <> ToString[NumberForm[Max[validAlphap], {4, 4}]] <> "] GeV⁻²\n";
      report = report <> "α' variation: " <> ToString[NumberForm[Max[validAlphap] - Min[validAlphap], {4, 4}]] <> " GeV⁻²\n\n"
    ]
  ];
  
  (* Predictions with uncertainties *)
  report = report <> "2. MASS PREDICTIONS WITH UNCERTAINTIES\n";
  report = report <> StringRepeat["-", 50] <> "\n";
  
  Do[
    jVal = prediction["J"];
    mass = prediction["PredictedMassGeV"];
    uncertainty = prediction["MassUncertaintyGeV"];
    nSigma = prediction["NSigma"];
    
    If[NumberQ[mass],
      report = report <> "J = " <> ToString[jVal] <> ": M = " <> 
                ToString[NumberForm[mass, {3, 3}]] <> " ± " <> 
                ToString[NumberForm[uncertainty, {3, 3}]] <> " GeV (" <> 
                ToString[nSigma] <> "σ)\n";
      report = report <> "  Prediction interval: [" <> 
                ToString[NumberForm[prediction["MassLowerGeV"], {3, 3}]] <> ", " <> 
                ToString[NumberForm[prediction["MassUpperGeV"], {3, 3}]] <> "] GeV\n",
      report = report <> "J = " <> ToString[jVal] <> ": Prediction failed\n"
    ],
    {prediction, predictions}
  ];
  report = report <> "\n";
  
  (* Cross-check results *)
  report = report <> "3. PDG CROSS-CHECK AND GAP ANALYSIS\n";
  report = report <> StringRepeat["-", 50] <> "\n";
  
  significantGaps = Select[crossCheckResults, #["IsSignificantGap"] &];
  
  report = report <> "Total predictions: " <> ToString[Length[crossCheckResults]] <> "\n";
  report = report <> "Significant gaps: " <> ToString[Length[significantGaps]] <> "\n\n";
  
  Do[
    jVal = row["J"];
    predMass = row["PredictedMassGeV"];
    uncertainty = row["MassUncertaintyGeV"];
    gapAnalysis = row["GapAnalysis"];
    
    report = report <> "J = " <> ToString[jVal] <> ": " <> gapAnalysis <> "\n";
    report = report <> "  Predicted: " <> ToString[NumberForm[predMass, {3, 3}]] <> " ± " <> 
              ToString[NumberForm[uncertainty, {3, 3}]] <> " GeV\n";
    report = report <> "  Search window: [" <> 
              ToString[NumberForm[row["SearchWindowLower"], {3, 3}]] <> ", " <> 
              ToString[NumberForm[row["SearchWindowUpper"], {3, 3}]] <> "] GeV\n";
    
    If[row["ClosestPDGEntry"] =!= None,
      closest = row["ClosestPDGEntry"];
      report = report <> "  Closest PDG: " <> closest["PDGEntry"] <> " at " <> 
                ToString[NumberForm[closest["PDGMassGeV"], {3, 3}]] <> " GeV (" <> 
                ToString[NumberForm[closest["DistanceSigma"], {3, 2}]] <> "σ)\n"
    ];
    report = report <> "\n",
    {row, significantGaps}
  ];
  
  (* Recommendations *)
  report = report <> "4. RECOMMENDATIONS\n";
  report = report <> StringRepeat["-", 50] <> "\n";
  
  report = report <> "Systematic Uncertainty Handling:\n";
  report = report <> "- Use κ = " <> ToString[errorAnalysisConfig["defaultKappa"]] <> 
            " as default width systematic fraction\n";
  report = report <> "- Include width-based systematic in all uncertainty calculations\n";
  report = report <> "- Propagate uncertainties through all predictions\n\n";
  
  report = report <> "Prediction Reliability:\n";
  report = report <> "- Predictions with large uncertainties should be treated with caution\n";
  report = report <> "- Significant gaps (no PDG entries within 2σ) indicate potential discoveries\n";
  report = report <> "- Cross-check predictions with theoretical expectations\n\n";
  
  report = report <> "=" <> StringRepeat["-", 78] <> "\n";
  
  (* Save report *)
  Export[outputFile, report, "Text"];
  report
];

(* 7. Main analysis function *)

(* Run comprehensive error analysis *)
runErrorAnalysis[reggeData_, fitFunction_, fitParams_, jValuesToPredict_, 
                pdgData_, outputDir_: "error_analysis"] := Module[
  {results = <||>},
  
  Print["=" <> StringRepeat["-", 58]];
  Print["SYSTEMATIC UNCERTAINTY ANALYSIS FOR REGGE TRAJECTORIES"];
  Print["=" <> StringRepeat["-", 58]];
  
  (* 1. Compute systematic uncertainties *)
  Print["1. Computing systematic uncertainties..."];
  dataWithUncertainties = computeSystematicUncertainties[reggeData, errorAnalysisConfig["defaultKappa"]];
  results["dataWithUncertainties"] = dataWithUncertainties;
  
  (* 2. Perform kappa sensitivity analysis *)
  Print["2. Performing kappa sensitivity analysis..."];
  sensitivityResults = analyzeKappaSensitivity[reggeData, fitFunction, fitParams];
  results["sensitivityAnalysis"] = sensitivityResults;
  
  (* 3. Fit with systematic uncertainties *)
  Print["3. Fitting with systematic uncertainties..."];
  fitResults = NonlinearModelFit[Transpose[{dataWithUncertainties[[All, "M2GeV2"]], 
                                           dataWithUncertainties[[All, "J"]]}], 
                                fitFunction[x, ##] & @@ fitParams, 
                                fitParams, x, 
                                Weights -> 1 / dataWithUncertainties[[All, "M2SigmaGeV2"]]^2];
  results["fitResults"] = fitResults;
  
  (* 4. Propagate uncertainties to predictions *)
  Print["4. Propagating uncertainties to predictions..."];
  predictions = propagateUncertaintiesToPredictions[
    fitResults["BestFitParameters"],
    fitResults["ParameterCovarianceMatrix"],
    jValuesToPredict,
    fitFunction
  ];
  results["predictions"] = predictions;
  
  (* 5. Cross-check with PDG *)
  Print["5. Cross-checking predictions with PDG..."];
  crossCheckResults = crossCheckPredictionsWithPDG[Normal[predictions], pdgData];
  results["crossCheckResults"] = crossCheckResults;
  
  (* 6. Generate visualizations and reports *)
  Print["6. Generating visualizations and reports..."];
  sensitivityPlot = createSensitivityPlots[sensitivityResults, outputDir];
  Export[FileNameJoin[{outputDir, "kappa_sensitivity_analysis.png"}], sensitivityPlot];
  
  report = generateErrorAnalysisReport[sensitivityResults, Normal[predictions], 
                                     Normal[crossCheckResults], 
                                     FileNameJoin[{outputDir, "error_analysis_report.txt"}]];
  
  Print["=" <> StringRepeat["-", 58]];
  Print["ERROR ANALYSIS COMPLETE!");
  Print["=" <> StringRepeat["-", 58]];
  
  results
];

(* 8. Example usage *)

(* Define linear fit function *)
linearFitFunction[x_, alpha0_, alphap_] := alpha0 + alphap * x;

(* Run analysis on your data *)
(* results = runErrorAnalysis[yourReggeData, linearFitFunction, {"alpha0", "alphap"}, 
                            {1.5, 2.5, 3.5, 4.5}, yourPDGData]; *)

(* Export results for Python comparison *)
(* Export["error_analysis_results.json", results]; *)
'''

# =============================================================================
# EXPECTED OUTPUT FORMATS
# =============================================================================

EXPECTED_ERROR_ANALYSIS_OUTPUT = """
Expected error_analysis_results.json format:
{
  "dataWithUncertainties": [
    {
      "MassGeV": 1.232,
      "J": 1.5,
      "M2GeV2": 1.518,
      "MassSigmaGeV": 0.002,
      "ResonanceWidthGeV": 0.118,
      "SystematicErrorGeV": 0.0295,
      "TotalUncertaintyGeV": 0.0296,
      "M2SigmaGeV2": 0.073,
      "Kappa": 0.25
    }
  ],
  "sensitivityAnalysis": {
    "kappaValues": [0.0, 0.025, 0.05, ...],
    "fitParameters": {
      "alpha0": [0.123, 0.124, 0.125, ...],
      "alphap": [0.890, 0.891, 0.892, ...]
    },
    "fitUncertainties": {
      "alpha0": [0.045, 0.046, 0.047, ...],
      "alphap": [0.012, 0.013, 0.014, ...]
    },
    "chi2Values": [1.23, 1.22, 1.21, ...],
    "rSquaredValues": [0.987, 0.988, 0.989, ...]
  },
  "fitResults": {
    "BestFitParameters": [0.124, 0.891],
    "ParameterCovarianceMatrix": [[0.002025, -0.00054], [-0.00054, 0.000144]],
    "ChiSquared": 12.3,
    "DegreesOfFreedom": 10,
    "RSquared": 0.988
  },
  "predictions": [
    {
      "J": 1.5,
      "PredictedMassGeV": 1.245,
      "MassUncertaintyGeV": 0.032,
      "MassLowerGeV": 1.213,
      "MassUpperGeV": 1.277,
      "PredictedM2GeV2": 1.550,
      "M2UncertaintyGeV2": 0.080,
      "ConfidenceLevel": 0.68,
      "NSigma": 1.0
    }
  ],
  "crossCheckResults": [
    {
      "J": 1.5,
      "PredictedMassGeV": 1.245,
      "MassUncertaintyGeV": 0.032,
      "SearchWindowLower": 1.181,
      "SearchWindowUpper": 1.309,
      "NearbyPDGEntries": [],
      "ClosestPDGEntry": null,
      "ClosestDistanceSigma": Infinity,
      "IsSignificantGap": true,
      "GapAnalysis": "No PDG entries found within 2σ window"
    }
  ]
}
"""

# =============================================================================
# VALIDATION FUNCTIONS (Python side)
# =============================================================================

def validate_error_analysis_results(json_path: str) -> bool:
    """
    Validate that exported error analysis results meet expected format.
    
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
    
    required_keys = ['dataWithUncertainties', 'sensitivityAnalysis', 
                    'fitResults', 'predictions', 'crossCheckResults']
    
    for key in required_keys:
        if key not in results:
            print(f"Missing key: {key}")
            return False
    
    # Check data with uncertainties
    data = results['dataWithUncertainties']
    if not isinstance(data, list) or len(data) == 0:
        print("Invalid dataWithUncertainties")
        return False
    
    required_data_fields = ['MassGeV', 'J', 'M2GeV2', 'SystematicErrorGeV', 
                           'TotalUncertaintyGeV', 'M2SigmaGeV2', 'Kappa']
    
    for field in required_data_fields:
        if field not in data[0]:
            print(f"Missing data field: {field}")
            return False
    
    # Check sensitivity analysis
    sensitivity = results['sensitivityAnalysis']
    sensitivity_keys = ['kappaValues', 'fitParameters', 'fitUncertainties', 
                       'chi2Values', 'rSquaredValues']
    
    for key in sensitivity_keys:
        if key not in sensitivity:
            print(f"Missing sensitivity key: {key}")
            return False
    
    # Check fit results
    fit_results = results['fitResults']
    fit_keys = ['BestFitParameters', 'ParameterCovarianceMatrix', 'ChiSquared', 
               'DegreesOfFreedom', 'RSquared']
    
    for key in fit_keys:
        if key not in fit_results:
            print(f"Missing fit result key: {key}")
            return False
    
    # Check predictions
    predictions = results['predictions']
    if not isinstance(predictions, list):
        print("Invalid predictions format")
        return False
    
    if len(predictions) > 0:
        pred_keys = ['J', 'PredictedMassGeV', 'MassUncertaintyGeV', 
                    'MassLowerGeV', 'MassUpperGeV', 'ConfidenceLevel', 'NSigma']
        
        for key in pred_keys:
            if key not in predictions[0]:
                print(f"Missing prediction field: {key}")
                return False
    
    # Check cross-check results
    cross_check = results['crossCheckResults']
    if not isinstance(cross_check, list):
        print("Invalid cross-check format")
        return False
    
    if len(cross_check) > 0:
        cross_check_keys = ['J', 'PredictedMassGeV', 'MassUncertaintyGeV', 
                           'IsSignificantGap', 'GapAnalysis']
        
        for key in cross_check_keys:
            if key not in cross_check[0]:
                print(f"Missing cross-check field: {key}")
                return False
    
    print("Error analysis results validation passed!")
    return True

def compare_wl_python_error_analysis(wl_json_path: str, python_results: dict) -> dict:
    """
    Compare Wolfram Language and Python error analysis results.
    
    Parameters:
    -----------
    wl_json_path : str
        Path to WL error analysis results JSON
    python_results : dict
        Python error analysis results
    
    Returns:
    --------
    dict
        Comparison results
    """
    import json
    
    with open(wl_json_path, 'r') as f:
        wl_results = json.load(f)
    
    comparison = {}
    
    # Compare fit parameters
    if 'fitResults' in wl_results and 'fit_results' in python_results:
        wl_fit = wl_results['fitResults']
        py_fit = python_results['fit_results']
        
        wl_params = wl_fit['BestFitParameters']
        py_params = py_fit['parameters']
        
        comparison['fit_parameters'] = {
            'wl_alpha0': wl_params[0],
            'python_alpha0': py_params[0],
            'wl_alphap': wl_params[1],
            'python_alphap': py_params[1],
            'alpha0_diff': abs(wl_params[0] - py_params[0]),
            'alphap_diff': abs(wl_params[1] - py_params[1])
        }
        
        # Compare chi-squared
        comparison['fit_quality'] = {
            'wl_chi2_dof': wl_fit['ChiSquared'] / wl_fit['DegreesOfFreedom'],
            'python_chi2_dof': py_fit['chi2_dof'],
            'wl_r_squared': wl_fit['RSquared'],
            'python_r_squared': py_fit['r_squared']
        }
    
    # Compare predictions
    if 'predictions' in wl_results and 'predictions' in python_results:
        wl_preds = wl_results['predictions']
        py_preds = python_results['predictions']
        
        if len(wl_preds) > 0 and len(py_preds) > 0:
            # Compare first prediction
            wl_pred = wl_preds[0]
            py_pred = py_preds.iloc[0]
            
            comparison['predictions'] = {
                'wl_predicted_mass': wl_pred['PredictedMassGeV'],
                'python_predicted_mass': py_pred['PredictedMassGeV'],
                'wl_mass_uncertainty': wl_pred['MassUncertaintyGeV'],
                'python_mass_uncertainty': py_pred['MassUncertaintyGeV'],
                'mass_diff': abs(wl_pred['PredictedMassGeV'] - py_pred['PredictedMassGeV']),
                'uncertainty_diff': abs(wl_pred['MassUncertaintyGeV'] - py_pred['MassUncertaintyGeV'])
            }
    
    # Compare systematic uncertainties
    if 'dataWithUncertainties' in wl_results and 'data_with_uncertainties' in python_results:
        wl_data = wl_results['dataWithUncertainties']
        py_data = python_results['data_with_uncertainties']
        
        if len(wl_data) > 0 and len(py_data) > 0:
            wl_sys_error = wl_data[0]['SystematicErrorGeV']
            py_sys_error = py_data.iloc[0]['SystematicErrorGeV']
            
            comparison['systematic_uncertainties'] = {
                'wl_systematic_error': wl_sys_error,
                'python_systematic_error': py_sys_error,
                'systematic_error_diff': abs(wl_sys_error - py_sys_error)
            }
    
    # Compare gap analysis
    if 'crossCheckResults' in wl_results and 'cross_check_results' in python_results:
        wl_gaps = wl_results['crossCheckResults']
        py_gaps = python_results['cross_check_results']
        
        wl_significant_gaps = sum(1 for gap in wl_gaps if gap['IsSignificantGap'])
        py_significant_gaps = len(py_gaps[py_gaps['IsSignificantGap']])
        
        comparison['gap_analysis'] = {
            'wl_significant_gaps': wl_significant_gaps,
            'python_significant_gaps': py_significant_gaps,
            'gap_count_agreement': wl_significant_gaps == py_significant_gaps
        }
    
    return comparison

if __name__ == "__main__":
    print("=" * 60)
    print("WOLFRAM LANGUAGE ERROR ANALYSIS GUIDE")
    print("=" * 60)
    print("\nAdd the following code to your Wolfram Language notebook:")
    print(ERROR_ANALYSIS_CODE)
    print("\nExpected output format:")
    print(EXPECTED_ERROR_ANALYSIS_OUTPUT)
    print("\nThis will provide:")
    print("1. Width-based systematic uncertainty handling (κ parameter)")
    print("2. Sensitivity analysis of α' vs κ")
    print("3. Uncertainty propagation to mass predictions")
    print("4. Cross-check with PDG data and gap analysis")
    print("5. Comprehensive error analysis reporting")
    print("6. Integration with existing Regge analysis pipeline")
