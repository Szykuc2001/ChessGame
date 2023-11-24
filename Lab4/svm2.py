import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

'''Read the dataset'''
data = pd.read_csv('dataset2.csv')

'''Apply LabelEncoder to the target variable type'''
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])  # Encoding the target variable
data['Chart?'] = label_encoder.fit_transform(data['Chart?'])  # Encoding the target variable
data['entity'] = label_encoder.fit_transform(data['entity'])  # Encoding the target variable
data['Banana values'] = label_encoder.fit_transform(data['Banana values'])  # Encoding the target variable

'''Data preprocessing: Split features and target variable'''
X = data.drop(['type', 'Unnamed: 16'], axis=1)
y = data['type']

'''Define the list of categorical columns to encode strings'''
categorical_cols = ['entity', 'Chart?']

'''Separating categorical and numerical columns'''
numerical_cols = X.select_dtypes(include='number').columns.tolist()

'''Apply LabelEncoder to categorical columns to encode strings'''
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

'''SimpleImputer is created with strategy='mean', it's set up to replace any missing (NaN) values in the dataset's 
numerical columns with the mean value of each column during the imputation process.'''
imputer = SimpleImputer(strategy='mean')  # You can use other strategies like median or mode

'''uses the SimpleImputer object (imputer) to handle missing values specifically in the numerical columns of the 
dataset X by replacing them with the mean values of their respective columns.'''
X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

'''20% of the data will be used for testing, and the remaining 80% will be for training.
random_state=42: This parameter ensures reproducibility by fixing the random seed. It ensures that each time the code 
is ran with the same random_state, the data split will be the same.
This separation of data into training and testing subsets allows the machine learning model to be trained on one 
portion of the data (X_train, y_train) and evaluated on another unseen portion (X_test, y_test) to assess its performance.'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Linear kernel computes the decision boundary as a straight line in the input space.
svm_classifier = SVC(kernel='linear', C=1.0) creates an SVM classifier object with a linear kernel and sets the 
regularization parameter C to 1.0. This classifier will aim to find a linear decision boundary in the feature space while 
balancing the margin maximization and error minimization based on the provided C value.'''
svm_classifier = SVC(kernel='linear', C=1.0)  # You can try different kernels and parameters

'''X_train: This is the feature matrix (training data) used to train the model. It contains a subset (80%) of the original 
dataset's features.
y_train: This is the target variable (training labels) associated with the X_train data. It contains the corresponding 
subset (80%) of the original dataset's target values.

The SVM classifier is trained using the provided training data (X_train) and their corresponding labels (y_train).
The classifier learns patterns and relationships between features and labels in the training data to create a decision
 boundary that separates different classes based on the input features.'''
svm_classifier.fit(X_train, y_train)

'''The trained SVM classifier (svm_classifier) applies the learned patterns and decision boundary to the features in the 
test set (X_test).
It predicts the target variable values (classification labels) for the test set based on the learned relationships 
between features and labels from the training data.'''
y_pred = svm_classifier.predict(X_test)

'''Evaluate the classifier'''
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

'''It compares the true labels (y_test) with the predicted labels (y_pred) and computes various metrics to evaluate 
the model's performance.
The report includes metrics such as precision, recall, F1-score, and support for each class present in the test set.
Precision: The ratio of correctly predicted positive observations to the total predicted positives.
Recall: The ratio of correctly predicted positive observations to the actual positives in the dataset.
F1-score: The harmonic mean of precision and recall, providing a balance between the two metrics.
Support: The number of actual occurrences of the class in the test set.'''
print(classification_report(y_test, y_pred))

'''PCA(n_components=2): This initializes a PCA object from Scikit-learn with the parameter n_components set to 2. 
It indicates that PCA will reduce the original feature space to two principal components.

pca.fit_transform(X): The fit_transform() method in PCA performs two main actions:

Fit: The fit() method in PCA analyzes the dataset (X) and calculates the principal components based on its features.

Transform: The transform() method then transforms the original dataset (X) into a new dataset (X_pca) using the 
calculated principal components. It projects the original data onto the lower-dimensional space defined by these 
principal components.'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

'''plt.figure(figsize=(8, 6)): This line initializes a new figure with a specific size of 8x6 inches for plotting.

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k'):

plt.scatter(): This function creates a scatter plot.
X_pca[:, 0] and X_pca[:, 1]: These represent the values of the transformed dataset (X_pca) along the first and second 
principal components, respectively, for the x and y axes of the plot.
c=y: This parameter colors the points based on the target variable (y), associating different colors with different 
classes or values in the target variable.
cmap='coolwarm': This parameter sets the colormap to be used for coloring the points. In this case, 'coolwarm' 
is the colormap chosen.
edgecolor='k': This parameter sets the edge color of the markers to black ('k') for better visibility.
plt.xlabel('Principal Component 1') and plt.ylabel('Principal Component 2'): These functions set labels for the x and y 
axes, indicating the principal components being visualized.

plt.title('SVM Classification (2D PCA)'): This function sets the title of the plot to 'SVM Classification (2D PCA)'.

plt.show(): This command displays the plot. Once executed, it shows the scatter plot representing the transformed data 
in a 2D space, where each point corresponds to a data sample, and the colors represent different classes or categories 
as defined by the target variable y. This visualization helps to understand the distribution and separability of the 
data in a reduced-dimensional space obtained through PCA.'''
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVM Classification (2D PCA)')
plt.show()