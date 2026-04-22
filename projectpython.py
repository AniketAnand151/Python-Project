import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (shapiro, ttest_ind, chi2_contingency, norm)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Data Loading and First Look
df = pd.read_csv(r"C:\Users\anike\OneDrive\Desktop\mai python hoon\CrimeData.csv")
df.info()
#describe
print(df.describe())
#print shape
print("Shape: ",df.shape)

#first 5 rows
print(df.head())
#check null and duplicate values
print("Null Values: \n", df.isnull().sum())
print("No. of Duplicate rows: ", df.duplicated().sum())

#Data Cleaning
df.rename(columns={
    'Area_Name': 'Area',
    'Year': 'Year',
    'Group_Name': 'Group',
    'Sub_Group_Name': 'SubGroup',

    'Cases_Acquitted_or_Discharged': 'Acquitted',
    'Cases_charge_sheets_were_not_laid_but_Final_Report_submitted': 'No_Chargesheet',
    'Cases_Chargesheeted': 'Chargesheeted',
    'Cases_Compounded_or_Withdrawn': 'Withdrawn',
    'Cases_Convicted': 'Convicted',
    'Cases_Declared_False_on_Account_of_Mistake_of_Fact_or_of_Law': 'False_Cases',
    'Cases_Investigated_Chargesheets+FR_Submitted': 'Investigated',
    'Cases_not_Investigated_or_in_which_investigation_was_refused': 'Not_Investigated',
    'Cases_Pending_Investigation_at_Year_End': 'Pending_Inv_End',
    'Cases_Pending_Investigation_from_previous_year': 'Pending_Inv',
    'Cases_Pending_Trial_at_Year_End': 'Pending_Trial_End',
    'Cases_Pending_Trial_from_the_previous_year': 'Pending_Trial',
    'Cases_Reported': 'Reported',
    'Cases_Sent_for_Trial': 'Trial',
    'Cases_Trials_Completed': 'Trials_Completed',
    'Cases_Withdrawn_by_the_Govt': 'Gov_Withdrawn',
    'Cases_withdrawn_by_the_Govt_during_investigation': 'Gov_Withdrawn_Inv',
    'Total_Cases_for_Trial': 'Total_Trial'
}, inplace=True)

# Print to confirm
print("Clean Column Names:")
for col in df.columns:
    print(col)

# Correlation Heatmap

num_cols = [
    'Reported',
    'Convicted',
    'Acquitted',
    'Pending_Inv',
    'Pending_Trial'
]

plt.figure(figsize=(8,5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap — Key Crime Variables")
plt.show()

#Distribution & Outlier Analysis of Cases Reported

# 1. Group-wise total reported cases
group_reported = df.groupby("Group")["Reported"].sum().sort_values(ascending=False)
print("Total Reported Cases by Crime Group:")
print(group_reported)

# Order for all graphs
group_order = group_reported.index

# 2. Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="Group", y="Reported", order=group_order)

plt.title("Reported Cases Distribution by Crime Group", fontsize=12)
plt.xlabel("Crime Group", fontsize=10)
plt.ylabel("Reported Cases", fontsize=10)
plt.xticks(rotation=30, ha="right", fontsize=8)

plt.tight_layout()
plt.show()

# 3. Countplot for high reported cases
high_reported = df[df["Reported"] > df["Reported"].median()]

plt.figure(figsize=(12,6))
sns.countplot(
    data=high_reported,
    x="Group",
    order=high_reported["Group"].value_counts().index,
    hue="Group",
    palette="Reds",
    legend=False
)

plt.title("Number of High Reported Cases by Crime Group", fontsize=12)
plt.xlabel("Crime Group", fontsize=10)
plt.ylabel("Count", fontsize=10)
plt.xticks(rotation=30, ha="right", fontsize=8)

plt.tight_layout()
plt.show()

# 4. Reported vs Convicted comparison
group_compare = df.groupby("Group")[["Reported", "Convicted"]].sum().loc[group_order]

group_compare.plot(kind="bar", figsize=(12,6), colormap="coolwarm")

plt.title("Reported vs Convicted Cases by Crime Group", fontsize=12)
plt.xlabel("Crime Group", fontsize=10)
plt.ylabel("Total Cases", fontsize=10)
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.legend(["Reported", "Convicted"], fontsize=9)

plt.tight_layout()
plt.show()



# State-wise & Year-wise Crime Trends


# 2a. Top 15 states by total cases reported
state_reported = df.groupby("Area")["Reported"].sum().sort_values(ascending=False).head(15)
print("Top 15 States by Total Reported Cases:")
print(state_reported)

plt.figure(figsize=(12,6))
sns.barplot(
    x=state_reported.values,
    y=state_reported.index,
    hue=state_reported.index,
    palette="Reds",
    legend=False
)

plt.title("Top 15 States by Total Cases Reported", fontsize=12)
plt.xlabel("Total Reported Cases", fontsize=10)
plt.ylabel("State", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()


# 2b. Year-wise national trend of cases reported
year_reported = df.groupby("Year")["Reported"].sum().sort_index()
print("\nYear-wise National Trend of Reported Cases:")
print(year_reported)

plt.figure(figsize=(10,5))
plt.plot(year_reported.index, year_reported.values, marker="o", linewidth=2)

plt.title("Year-wise National Trend of Cases Reported", fontsize=12)
plt.xlabel("Year", fontsize=10)
plt.ylabel("Total Reported Cases", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# 2c. Cases Sent for Trial vs Cases Convicted for top 10 states
# Top 10 states
top10_states = df.groupby("Area")["Reported"].sum().sort_values(ascending=False).head(10).index

# Data for comparison
trial_convicted = df[df["Area"].isin(top10_states)].groupby("Area")[["Trial", "Convicted"]].sum()

# X positions
import numpy as np
x = np.arange(len(trial_convicted.index))
width = 0.35

plt.figure(figsize=(12,6))

# Bars side by side
plt.bar(x - width/2, trial_convicted["Trial"], width, label="Sent for Trial")
plt.bar(x + width/2, trial_convicted["Convicted"], width, label="Convicted")

# Labels
plt.title("Cases Sent for Trial vs Cases Convicted (Top 10 States)", fontsize=12)
plt.xlabel("State", fontsize=10)
plt.ylabel("Total Cases", fontsize=10)

plt.xticks(x, trial_convicted.index, rotation=30, ha="right", fontsize=8)
plt.legend()

plt.tight_layout()
plt.show()

# Crime Group & Sub-Group Analysis
#pie chart
group_cases = df.groupby("Group")["Reported"].sum().sort_values(ascending=False)

top6 = group_cases.head(6)
others = pd.Series({"Others": group_cases.iloc[6:].sum()})

final_data = pd.concat([top6, others])

plt.figure(figsize=(7,7))
plt.pie(
    final_data.values,
    labels=[i.split(" - ")[0] for i in final_data.index],  # shorter names
    autopct="%1.1f%%",
    startangle=140
)

plt.title("Crime Group Share (Simplified)", fontsize=12)
plt.tight_layout()
plt.show()


# 3b. Horizontal bar — Top 10 sub-groups by total cases reported
subgroup_cases = df.groupby("SubGroup")["Reported"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Sub-Groups by Total Reported Cases:")
print(subgroup_cases)

plt.figure(figsize=(12,6))
sns.barplot(
    x=subgroup_cases.values,
    y=subgroup_cases.index,
    hue=subgroup_cases.index,
    palette="viridis",
    legend=False
)

plt.title("Top 10 Sub-Groups by Total Reported Cases", fontsize=12)
plt.xlabel("Total Reported Cases", fontsize=10)
plt.ylabel("Sub-Group", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# =========================================
# Objective 4
# Hypothesis Testing — Does Chargesheeting Rate Affect Cases Reported?
# =========================================

# ==============================
# DESCRIPTIVE STATISTICS
# ==============================

cols = ["Reported", "Chargesheeted", "Convicted", "Acquitted", "Trial"]

print("DESCRIPTIVE STATISTICS")
print(df[cols].describe().round(2))


# ==============================
# T-test
# H0: High and Low chargesheeted groups have same average reported cases
# H1: High chargesheeted group has significantly different reported cases
# ==============================

# Create groups
median_charge = df["Chargesheeted"].median()

high_group = df[df["Chargesheeted"] > median_charge]["Reported"]
low_group  = df[df["Chargesheeted"] <= median_charge]["Reported"]

# Perform T-test
t_stat, p_t = ttest_ind(high_group, low_group)

print("=" * 55)
print(" T-test: High Chargesheeted VS Low Chargesheeted")
print("=" * 55)

print(f" High Chargesheeted Mean : {high_group.mean():.2f}")
print(f" Low Chargesheeted Mean  : {low_group.mean():.2f}")
print(f" Difference              : {high_group.mean() - low_group.mean():.2f}")

print(f" T-statistic : {t_stat:.4f}")
print(f" p-value     : {p_t:.4f}")
print()

# Decision
if p_t < 0.05:
    print(" Reject H0: Chargesheeting significantly affects reported cases")
else:
    print(" Cannot Reject H0: No significant difference found")
# Objective 5
# Simple Linear Regression — Predicting Cases Reported
# =========================================

# 5a. Pre-model scatter — Chargesheeted vs Reported
plt.figure(figsize=(8,5))
plt.scatter(df["Chargesheeted"], df["Reported"], alpha=0.4, color="steelblue")
plt.title("Cases Chargesheeted vs Cases Reported", fontsize=12)
plt.xlabel("Cases Chargesheeted", fontsize=10)
plt.ylabel("Cases Reported", fontsize=10)
plt.tight_layout()
plt.show()


# Train-test split (80/20)
X = df[["Chargesheeted"]]
y = df["Reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set : {X_train.shape[0]} rows")
print(f"Testing set  : {X_test.shape[0]} rows")


# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("=" * 55)
print("  Linear Regression — Coefficients")
print("=" * 55)

slope = model.coef_[0]
intercept = model.intercept_

print(f"  Slope     : {slope:.4f}")
print(f"  Intercept : {intercept:.4f}")
print(f"\n Equation : Reported = {slope:.4f} * Chargesheeted + {intercept:.4f}")


# Evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=" * 35)
print("  Model Evaluation on Test Set")
print("=" * 35)
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.4f}")
print(f"  RMSE     : {rmse:.4f}")


# 5b. Actual vs Predicted scatter
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.4, color="steelblue")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=1.5
)

plt.title("Actual vs Predicted Cases Reported", fontsize=12)
plt.xlabel("Actual Reported Cases", fontsize=10)
plt.ylabel("Predicted Reported Cases", fontsize=10)
plt.tight_layout()
plt.show()


# 5c. Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.4, color="purple")
plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)

plt.title("Residual Plot", fontsize=12)
plt.xlabel("Predicted Reported Cases", fontsize=10)
plt.ylabel("Residuals", fontsize=10)
plt.tight_layout()
plt.show()


