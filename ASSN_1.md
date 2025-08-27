# Hands-On Data Wrangling Lab: Exploratory Data Analysis (EDA) for Real-World Datasets

## Lab Overview

**IMPORTANT - Read This First:**
This entire exercise is about **not getting stuck**. You have many resources available to you:
- **Your teammates** - collaborate and help each other
- **Google** - search for solutions to coding problems
- **LLMs like ChatGPT** - ask for help with code or concepts
- **Colab's built-in AI** - use the sparkle ✨ icon for AI assistance
- **The reference notebook** - use it as your guide

**Team Expectations:**
- If working in groups (max 4 people), **everyone must contribute** to the analysis
- Even though you work together, **everyone submits their own individual work**
- You don't have to be perfect or complete every single step
- **Use your judgment** - focus on producing a quality analysis within the time constraints

In this comprehensive data wrangling lab, you will perform a complete Exploratory Data Analysis (EDA) on a dataset of your choice. You'll follow industry-standard practices to investigate data structure, assess data quality, and uncover meaningful content insights that drive data-driven decision making.

**Learning Objectives:**
- Master the three pillars of EDA: Structure, Quality, and Content Investigation
- Apply Python libraries for comprehensive data analysis and visualization
- Identify and handle common data quality issues (duplicates, missing values, outliers)
- Generate actionable insights from data exploration
- Document findings in a professional format

**Prerequisites:**
- Basic Python programming knowledge
- Familiarity with Pandas and NumPy
- Understanding of basic statistics and data visualization concepts

**Use Case:**
You are a data analyst for a consulting firm tasked with evaluating a new client's dataset. Your client needs to understand their data's reliability and discover hidden patterns before implementing any business intelligence solutions. Your EDA will inform critical business decisions and data pipeline design.

## Lab Environment

**Required Software:**
- Google Colab (recommended) or Jupyter Notebook
- Python 3.7+
- Internet connection for dataset access

**Required Libraries:**
- pandas, numpy, matplotlib, seaborn
- sklearn, scipy.stats
- missingno (for missing data visualization)
- ssl, certifi (for secure data downloads)

**Hardware Requirements:**
- Minimum 4GB RAM
- Modern web browser
- Stable internet connection for dataset downloads

## Step-by-Step Instructions

### Step 1: Environment Setup and Library Installation
In this step, we'll prepare your Python environment with all necessary libraries for comprehensive data analysis.

Open Google Colab at https://colab.research.google.com/ and create a new notebook. Install and import the required libraries:

```python
# Install missing data visualization library
!pip install missingno

# Import essential libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import missingno as msno
import ssl
import certifi
from sklearn.datasets import fetch_openml
%matplotlib inline

# Configure SSL for secure downloads
ssl._create_default_https_context = ssl._create_unverified_context
```

This setup ensures you have all the tools needed for professional-grade exploratory data analysis, including specialized libraries for missing data visualization.

### Step 2: Dataset Selection and Acquisition
Choose your dataset strategically to ensure meaningful analysis results. You have several options for obtaining data:
**Option 0**: Just use the Stress_Dataset.csv file that is in this github repo

**Option A: Using Kaggle Datasets (Recommended)**

1. Visit **Kaggle**: https://www.kaggle.com/datasets 📊
2. Browse datasets and select one with at least 1000 rows and 10+ features
3. Click on your chosen dataset
4. Click the **Download Zip** button (you may need to create a free Kaggle account).  You must EXTRACT the zip first.  Find files that are .csv (not Excel)
5. **Upload to Colab:**

```python
# Method 1: Direct file upload (if dataset is small < 25MB)
from google.colab import files
uploaded = files.upload()  # This will open a file browser

# Get the filename of uploaded file
filename = list(uploaded.keys())[0]
print(f"Uploaded file: {filename}")

# Load the dataset
df_X = pd.read_csv(filename)
print(f"Dataset loaded successfully with shape: {df_X.shape}")
```

**Alternative Method for Larger Files:**
```python
# Method 2: Using Kaggle API (for larger datasets)
# First, install kaggle package
!pip install kaggle

# Upload your kaggle.json file (download from Kaggle Account settings)
from google.colab import files
files.upload()  # Upload kaggle.json

# Set up Kaggle API
!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset (replace with your chosen dataset)
!kaggle datasets download -d [username/dataset-name]
!unzip [dataset-name].zip

# Load the dataset
df_X = pd.read_csv('[your-csv-file].csv')
```

**Option B: Using OpenML**

For OpenML datasets, use the fetch_openml function:
```python
# Example: Load dataset by ID (replace with your chosen dataset ID)
dataset = fetch_openml(data_id=YOUR_DATASET_ID, as_frame=True)
df_X = dataset["frame"]
```

**Option C: Using Direct URLs**
```python
# If you have a direct URL to a CSV file
url = "https://your-dataset-url.csv"
df_X = pd.read_csv(url)
```

Document your dataset choice and source in a markdown cell, explaining why this dataset is relevant to your analysis goals.

### Step 3: Initial Data Structure Investigation
Understand the fundamental characteristics of your dataset before diving deeper.

Execute comprehensive structural analysis:
```python
# Display basic information about your dataset
print(f"Dataset shape: {df_X.shape}")
print(f"Dataset type: {type(df_X)}")

# Show random samples to understand data format
display(df_X.sample(5))

# Get detailed information about columns and data types
df_X.info()

# Count occurrences of each data type
print("Data type distribution:")
print(pd.value_counts(df_X.dtypes))
```

This step reveals the dataset's dimensionality, memory usage, and the mix of numerical versus categorical features, which informs your subsequent analysis approach.

### Step 4: Numerical vs Non-Numerical Feature Analysis
Categorize and analyze different feature types to apply appropriate analytical techniques.

Separate and analyze numerical features:
```python
# Analyze unique values in numerical features
unique_values = df_X.select_dtypes(include='number').nunique().sort_values()

# Create visualization of unique value distribution
plt.figure(figsize=(15, 4))
sns.set_style('whitegrid')
g = sns.barplot(x=unique_values.index, y=unique_values, palette='inferno')
g.set_yscale("log")
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_title('Unique Values per Numerical Feature (Log Scale)')

# Add value labels to bars
for index, value in enumerate(unique_values):
    g.text(index, value, str(value), color='black', ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()
```

Analyze non-numerical features:
```python
# Display non-numerical features structure
df_X.select_dtypes(exclude="number").describe()
```

This categorization helps you identify binary features (2 unique values), ordinal features (3-10 unique values), and continuous features (>10 unique values).

### Step 5: Data Quality Investigation - Duplicate Detection
Identify and handle duplicate records that could skew your analysis results.

```python
# Check for duplicate records (excluding index columns if present)
index_cols = [col for col in df_X.columns if 'index' in col.lower() or 'id' in col.lower()]
analysis_cols = [col for col in df_X.columns if col not in index_cols]

n_duplicates = df_X[analysis_cols].duplicated().sum()
print(f"Number of duplicate records: {n_duplicates}")
print(f"Percentage of duplicates: {(n_duplicates/len(df_X))*100:.2f}%")

# Remove duplicates if found
if n_duplicates > 0:
    df_X.drop_duplicates(subset=analysis_cols, inplace=True)
    print(f"Dataset shape after removing duplicates: {df_X.shape}")
```

Duplicate removal is crucial for accurate statistical analysis, as duplicate records can artificially inflate certain patterns and correlations in your data.

### Step 6: Data Quality Investigation - Missing Values Analysis
Systematically identify and address missing data patterns.

Visualize missing data patterns:
```python
# Create missing data heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(df_X.isnull(), cbar=True, cmap='viridis', 
            yticklabels=False, xticklabels=True)
plt.title('Missing Data Pattern Heatmap')
plt.xlabel('Features')
plt.ylabel('Records')
plt.xticks(rotation=45)
plt.show()

# Use missingno for advanced missing data visualization
msno.matrix(df_X, labels=True, sort='descending', color=(0.27, 0.52, 1.0))
plt.show()
```

Apply data cleaning strategies:
```python
# Remove rows with excessive missing values (>20% missing)
df_X = df_X.dropna(thresh=df_X.shape[1] * 0.80, axis=0).reset_index(drop=True)

# Remove columns with excessive missing values (>15% missing)  
df_X = df_X.dropna(thresh=df_X.shape[0] * 0.85, axis=1)

print(f"Dataset shape after missing value cleanup: {df_X.shape}")

# Visualize remaining missing data
df_X.isna().mean().sort_values().plot(
    kind="bar", figsize=(15, 4),
    title="Percentage of Missing Values per Feature (After Cleanup)",
    ylabel="Ratio of Missing Values")
plt.xticks(rotation=45)
plt.show()
```

### Step 7: Content Investigation - Feature Distribution Analysis
Explore the distribution characteristics of your features to understand data behavior.

```python
# Create histograms for all numerical features
df_X.hist(bins=25, figsize=(15, 25), layout=(-1, 5), edgecolor="black")
plt.suptitle('Distribution of All Numerical Features', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

# Identify features dominated by single values
most_frequent_entry = df_X.mode()
df_freq = df_X.eq(most_frequent_entry.values, axis=1)
df_freq = df_freq.mean().sort_values(ascending=False)

print("Top 5 features with highest single-value dominance:")
display(df_freq.head())

# Visualize single-value dominance
df_freq.plot.bar(figsize=(15, 4), 
                title="Single Value Dominance Across Features",
                ylabel="Proportion of Most Frequent Value")
plt.xticks(rotation=45)
plt.show()
```

This analysis reveals whether features have normal distributions, are skewed, or are dominated by single values, which impacts modeling decisions.

### Step 8: Content Investigation - Feature Relationships and Correlations
Discover relationships between features that could indicate redundancy or important patterns.

```python
# Separate continuous vs discrete features for appropriate analysis
cols_continuous = df_X.select_dtypes(include="number").nunique() >= 25
df_continuous = df_X[cols_continuous[cols_continuous].index]

# Create correlation matrix for continuous features
if len(df_continuous.columns) > 1:
    df_corr = df_continuous.corr(method="pearson")
    
    # Create correlation strength labels
    labels = np.where(np.abs(df_corr)>0.75, "S",  # Strong
             np.where(np.abs(df_corr)>0.5, "M",   # Medium  
             np.where(np.abs(df_corr)>0.25, "W", ""))) # Weak
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 12))
    sns.heatmap(df_corr, mask=np.eye(len(df_corr)), square=True,
                center=0, annot=labels, fmt='', linewidths=.5,
                cmap="RdBu_r", cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix\n(S=Strong >0.75, M=Medium >0.5, W=Weak >0.25)')
    plt.show()
    
    # Identify strongest correlations
    lower_triangle_mask = np.tril(np.ones(df_corr.shape), k=-1).astype("bool")
    df_corr_stacked = df_corr.where(lower_triangle_mask).stack().sort_values()
    
    print("Strongest positive and negative correlations:")
    print("Most Negative:")
    display(df_corr_stacked.head())
    print("\nMost Positive:")  
    display(df_corr_stacked.tail())
```

### Step 9: Advanced Content Investigation - Discrete Feature Analysis
Analyze categorical and discrete features for patterns and relationships.

```python
# Analyze discrete features
df_discrete = df_X[cols_continuous[~cols_continuous].index]

if len(df_discrete.columns) > 0:
    # Create strip plots for discrete features against a continuous target
    continuous_cols = df_continuous.columns
    if len(continuous_cols) > 0:
        target_col = continuous_cols[0]  # Use first continuous column as target
        
        # Setup subplot grid
        n_cols = 3
        n_elements = min(len(df_discrete.columns), 9)  # Limit to 9 plots
        n_rows = np.ceil(n_elements / n_cols).astype("int")
        
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, 
                               figsize=(15, n_rows * 3))
        
        # Create plots for each discrete feature
        for i, col in enumerate(df_discrete.columns[:n_elements]):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
            
            try:
                sns.stripplot(data=df_X, x=col, y=target_col, 
                            ax=ax, palette="Set2", size=2, alpha=0.6)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            except:
                ax.text(0.5, 0.5, f'Cannot plot\n{col}', 
                       transform=ax.transAxes, ha='center')
        
        plt.suptitle(f'Discrete Features vs {target_col}', fontsize=16)
        plt.tight_layout()
        plt.show()
```

### Step 10: Documentation and Insight Generation
Compile your findings into actionable insights and prepare deliverables.

Create comprehensive documentation:
```python
# Generate summary statistics report
print("=== EXPLORATORY DATA ANALYSIS SUMMARY ===")
print(f"Dataset Dimensions: {df_X.shape[0]:,} rows × {df_X.shape[1]} columns")
print(f"Memory Usage: {df_X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Numerical Features: {len(df_X.select_dtypes(include='number').columns)}")
print(f"Categorical Features: {len(df_X.select_dtypes(exclude='number').columns)}")
print(f"Missing Values: {df_X.isnull().sum().sum():,} total")
print(f"Complete Records: {len(df_X.dropna()):,} ({len(df_X.dropna())/len(df_X)*100:.1f}%)")

# Identify key insights for your 1-page writeup
print("\n=== KEY FINDINGS FOR WRITEUP ===")
print("1. Data Structure Insights:")
print("   - [Document your structural findings here]")
print("2. Data Quality Issues:")
print("   - [Document quality issues and how you addressed them]")
print("3. Content Patterns:")
print("   - [Document interesting patterns, correlations, distributions]")
print("4. Business Implications:")
print("   - [Connect technical findings to business value]")
```

**Bonus Challenge - Automated EDA Application:**
Create an automated EDA application using Bolt.new (https://bolt.new):

1. Visit **Bolt.new** 🚀
2. Create a web application that accepts CSV file uploads
3. Implement automated generation of:
   - Structure analysis dashboard
   - Missing data visualization
   - Correlation heatmaps  
   - Distribution plots
4. Add export functionality for analysis reports
5. Deploy your application and include the URL in your submission

## Conclusion

**What You've Accomplished:**
You have completed a comprehensive Exploratory Data Analysis following industry best practices. You've systematically investigated data structure, identified and resolved quality issues, and uncovered meaningful content patterns. These skills are fundamental to any data science project and directly applicable to real-world business scenarios.

**Key Takeaways:**
- EDA is a critical foundation for any data science project
- Systematic investigation (Structure → Quality → Content) ensures comprehensive understanding
- Visual analysis often reveals patterns not apparent in statistical summaries
- Data quality issues must be addressed before analysis can produce reliable insights

**Next Steps for Further Learning:**
- Explore advanced missing data imputation techniques
- Study feature engineering based on EDA findings
- Learn automated EDA tools like pandas-profiling and sweetviz
- Practice with domain-specific datasets relevant to your career interests

## Deliverables Checklist

✅ **Individual Submission Required** (even if working in groups):
1. **Jupyter Notebook** with complete EDA analysis
2. **1-Page Written Summary** including:
   - Dataset description and source
   - Key structural findings
   - Quality issues discovered and resolution approach
   - Most significant content insights
   - Business implications and recommendations
3. **Bonus**: Bolt.new application URL (extra credit)

**Remember:** You don't need to complete every single step perfectly. Use your judgment, focus on quality over quantity, and submit work that demonstrates your understanding of the EDA process.

## References

- Reference Notebook: https://colab.research.google.com/drive/1WP1-U7BlfHzq-374HA1vTITB9p6pxZl7?usp=sharing
- OpenML Dataset Repository: https://www.openml.org/search?type=data&status=active
- Kaggle Datasets: https://www.kaggle.com/datasets  
- Pandas Documentation: https://pandas.pydata.org/docs/
- Seaborn Visualization Gallery: https://seaborn.pydata.org/examples/index.html
- Missingno Documentation: https://github.com/ResidentMario/missingno

**Time Management Tip:** Allocate 15 minutes for setup and data loading, 45 minutes for analysis steps 3-9, and 15 minutes for documentation and insight generation to stay within the 75-minute timeframe.
