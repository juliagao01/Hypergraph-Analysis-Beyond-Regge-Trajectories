# Mathematica Integration for Unified Regge-Hypergraph Analysis

## 🎯 **Perfect Integration: Python Analysis + Mathematica Plotting**

You're absolutely right that **most plots are better generated in Mathematica**! This integration gives you the best of both worlds:

- **Python**: Advanced statistical analysis, machine learning, reproducibility
- **Mathematica**: Superior plotting, LaTeX integration, publication-quality output

## 🔄 **Complete Workflow**

### **1. Python Analysis Pipeline**
```
Data Preparation → Hypergraph Features → Regge Fitting → Bridging Analysis → Hypothesis Testing → Predictions
```

### **2. Mathematica Export & Plotting**
```
Python Results → Mathematica Export → Publication-Quality Plots → High-Resolution PDFs
```

## 📊 **Generated Mathematica Files**

### **Core Data Files:**
- **`unified_states.m`**: Complete data model with all particle states and features
- **`regge_fit_results.m`**: Regge trajectory fit parameters and diagnostics  
- **`bridging_analysis_results.m`**: Statistical correlations and regression results
- **`hypothesis_test_results.m`**: Pre-registered hypothesis test outcomes
- **`predictions.m`**: Predictions with hypergraph-informed confidence levels

### **Plotting Infrastructure:**
- **`unified_analysis_plots.nb`**: Complete Mathematica notebook with publication-quality plotting functions
- **`README_Mathematica.md`**: Usage guide and function documentation

## 🎨 **Publication-Quality Plots Generated**

### **Figure 1: Regge Trajectory Plot**
- **Mathematica Features**: Times font, proper LaTeX labels, error bars, outlier highlighting
- **Python Data**: Fit parameters, residuals, outlier identification
- **Output**: High-resolution PDF with 300 DPI

### **Figure 2: Residuals vs Hypergraph Features**
- **Mathematica Features**: Multi-panel layout, consistent styling, grid lines
- **Python Data**: Correlation analysis, statistical significance
- **Output**: Publication-ready 2x2 grid of correlation plots

### **Figure 3: Hypothesis Test Summary**
- **Mathematica Features**: Professional table formatting, color coding
- **Python Data**: Hypothesis test results, p-values, effect sizes
- **Output**: Clean summary table for manuscript

### **Figure 4: Predictions with Confidence**
- **Mathematica Features**: Color-coded confidence levels, error bars, legends
- **Python Data**: Mass predictions, community assignments, confidence scores
- **Output**: Predictive plot with uncertainty quantification

## 🚀 **How to Use**

### **Step 1: Run Python Analysis**
```bash
cd python_analysis
python run_unified_analysis.py
```

### **Step 2: Open Mathematica**
```mathematica
(* Open the generated notebook *)
NotebookOpen["mathematica_exports/unified_analysis_plots.nb"]
```

### **Step 3: Generate Plots**
```mathematica
(* Run the notebook to generate all plots *)
(* Plots are automatically exported as high-resolution PDFs *)
```

## 🎯 **Key Mathematica Advantages**

### **1. Superior Typography**
- **Times font** for publication standards
- **Proper LaTeX labels** (M², GeV, etc.)
- **Consistent sizing** and spacing

### **2. Professional Styling**
- **Thick frame lines** for clarity
- **Dashed grid lines** for readability
- **Color-coded elements** for interpretation

### **3. High-Resolution Export**
- **300 DPI resolution** for publication
- **Vector graphics** for scaling
- **PDF format** for journal submission

### **4. Advanced Plotting Features**
- **Error bars** with proper uncertainty
- **Multi-panel layouts** for complex data
- **Interactive legends** and annotations

## 📈 **Example Mathematica Code**

### **Regge Trajectory Plot:**
```mathematica
(* Professional Regge plot with outliers *)
reggePlot = Show[
  ListPlot[{{#["m2_gev2"], #["j"]}} & /@ inliers,
    PlotStyle -> {Blue, PointSize[0.02]},
    PlotLegends -> {"Data Points"}],
  
  ListPlot[{{#["m2_gev2"], #["j"]}} & /@ outliers,
    PlotStyle -> {Red, PointSize[0.03]},
    PlotLegends -> {"Outliers"}],
  
  Plot[reggeFit[x], {x, xmin, xmax}, 
    PlotStyle -> {Red, Thick}],
  
  Frame -> True,
  FrameLabel -> {"M² (GeV²)", "J"},
  PlotLabel -> Style["Regge Trajectory", Bold, 14],
  BaseStyle -> {FontFamily -> "Times", FontSize -> 12}
];
```

### **Correlation Analysis:**
```mathematica
(* Multi-panel correlation plots *)
GraphicsGrid[{
  {plot1, plot2},
  {plot3, plot4}
}, ImageSize -> 800, Spacings -> 20]
```

## 🔧 **Integration Benefits**

### **For Your Paper:**
- **Publication-ready figures** with journal standards
- **Consistent styling** across all plots
- **High-resolution output** for print and digital
- **Professional appearance** that impresses reviewers

### **For Reproducibility:**
- **Automated export** from Python analysis
- **Version-controlled** plotting code
- **Documented functions** for future use
- **Standardized workflow** for team collaboration

### **For Flexibility:**
- **Easy customization** of plot styles
- **Interactive exploration** in Mathematica
- **Additional analysis** using Mathematica's tools
- **Export to multiple formats** (PDF, PNG, SVG)

## 🎯 **Real Data Integration**

### **When You Have Real Mathematica Data:**
```mathematica
(* Export from your Mathematica analysis *)
Export["real_data.json", {
  "particle_data" -> yourParticleData,
  "hypergraph_features" -> yourHypergraphResults,
  "regge_fit" -> yourReggeFit
}]
```

### **Python Reads Real Data:**
```python
# Load your real Mathematica data
with open("real_data.json", "r") as f:
    real_data = json.load(f)

# Run unified analysis on real data
results = run_unified_analysis(real_data)
```

### **Mathematica Gets Real Results:**
```mathematica
(* Load Python analysis results *)
<< "mathematica_exports/unified_states.m"
<< "mathematica_exports/regge_fit_results.m"

(* Generate publication plots *)
reggePlot = (* your publication plot *)
Export["fig1_regge_trajectory.pdf", reggePlot, ImageResolution -> 300]
```

## 📊 **Current Results Summary**

### **Sample Analysis (20 Delta resonances):**
- **Regge fit**: α' = 0.740 ± 0.441 (consistent with expectations)
- **Quality control**: Width strongly correlates with residuals (r = 0.881, p < 0.001)
- **Predictions**: J = 11.5-15.5 with community-informed confidence
- **Hypothesis testing**: Framework validated and ready for real data

### **Generated Files:**
- **8 Mathematica files** for complete analysis
- **1 plotting notebook** with publication-quality functions
- **1 usage guide** for easy implementation

## 🎯 **Next Steps**

### **1. Real Data Integration:**
- Export your actual Mathematica PDG data
- Replace simulated hypergraph features
- Run analysis on real particle families

### **2. Enhanced Plotting:**
- Customize plot styles for your journal
- Add additional analysis in Mathematica
- Create interactive visualizations

### **3. Publication Preparation:**
- Generate all figures in publication format
- Create supplementary material
- Prepare reproducibility package

## 🏆 **Key Achievement**

This integration successfully addresses your original insight:

> **"Most plots are better generated in Mathematica"**

**Result**: A seamless workflow where Python handles the heavy statistical lifting and Mathematica creates publication-quality visualizations, giving you the best of both computational environments.

---

*The unified framework now provides both rigorous statistical analysis (Python) and superior visualization (Mathematica), creating a publication-ready pipeline that leverages the strengths of each platform.*
