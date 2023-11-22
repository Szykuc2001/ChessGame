import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv('dataset2.csv')

# Apply LabelEncoder to the target variable 'type'
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])  # Encoding the target variable
data['Chart?'] = label_encoder.fit_transform(data['Chart?'])  # Encoding the target variable
data['entity'] = label_encoder.fit_transform(data['entity'])  # Encoding the target variable
data['Banana values'] = label_encoder.fit_transform(data['Banana values'])  # Encoding the target variable


# Data preprocessing: Split features and target variable
X = data.drop(['type', 'Unnamed: 16'], axis=1)  # Assuming 'Unnamed: 16' is another column you want to drop
y = data['type']

# Define the list of categorical columns
categorical_cols = ['entity', 'Chart?']  # Replace with your categorical column names

# Separating categorical and numerical columns
numerical_cols = X.select_dtypes(include='number').columns.tolist()

# Apply LabelEncoder to categorical columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the list of numerical columns


# Define the imputer
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like median or mode

# Fit and transform the imputer on the training data
X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)  # You can try different kernels and parameters

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Additional evaluation metrics
print(classification_report(y_test, y_pred))