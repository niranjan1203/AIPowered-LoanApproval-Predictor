import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
  # Fixes incorrect column separation

# Load the training data with correct delimiter handling
df = pd.read_csv("C:/Users/Niranjan R M/Desktop/loan/loan_fixed.csv", delimiter=";") 

# Print first few rows
print("\n✅ First 5 Rows of Dataset:")
print(df.head())

# ✅ Debugging Step: Print unique values in Loan_Approved
print("\n✅ Unique values in Loan_Approved before conversion:", df['Loan_Approved'].unique())

# Convert Employment_Type to numeric (0 for Salaried, 1 for Self-Employed)
df['Employment_Type'] = df['Employment_Type'].map({"Salaried": 0, "Self-Employed": 1})

# ✅ Fix Loan_Approved conversion (handle unknown values)
valid_values = {"Approved": 1, "Rejected": 0}
df['Loan_Approved'] = df['Loan_Approved'].map(valid_values)

# ✅ Drop rows where Loan_Approved is NaN (fix unexpected values)
df = df.dropna(subset=['Loan_Approved'])

# ✅ Debugging: Print the number of remaining rows
print("\n✅ Number of rows after cleaning:", df.shape[0])

# If dataset is empty, raise error
if df.shape[0] == 0:
    raise ValueError("❌ ERROR: No valid data left! Check CSV file for unexpected Loan_Approved values.")

# Define features & target
X = df[['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Type']]
y = df['Loan_Approved']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
joblib.dump(model, "loan_approval_model.pkl")

print("✅ Model training complete! Saved as 'loan_approval_model.pkl'")