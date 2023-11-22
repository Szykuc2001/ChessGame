import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace 'your_dataset.csv' with your actual file path)
df = pd.read_csv('dataset2.csv')

# Replace 'Cows milk' with a standard category label in the 'entity' column
df['entity'].replace('Cows milk', 'Other', inplace=True)

# Assuming 'type' is categorical and needs encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['type'])  # Encoding the target variable

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical columns for features
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

# Drop the original categorical columns from the features
df.drop(categorical_cols, axis=1, inplace=True)

# Features after encoding and dropping original categorical columns
X = df.drop(['type'], axis=1)  # Features

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')