#Link to used dataset: https://www.kaggle.com/datasets/joebeachcapital/banana-index

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz

'''Read the dataset'''
df = pd.read_csv('dataset2.csv')

'''df['entity']: Accesses the 'entity' column in the DataFrame df.

.replace('Cows milk', 'Other', inplace=True): This function replaces occurrences of the value 'Cows milk' with 'Other' 
within the 'entity' column.

'Cows milk': The value to be replaced.
'Other': The value that will replace 'Cows milk'.
inplace=True: This parameter specifies that the modification should be made directly to the DataFrame df rather 
than returning a modified copy. When inplace=True, the DataFrame is updated with the changes.'''
df['entity'].replace('Cows milk', 'Other', inplace=True)

'''LabelEncoder is employed to convert categorical labels into numerical format, 
facilitating their use in machine learning models that require numerical inputs for training and prediction. 
The fit_transform method both fits the encoder to the unique categories present in the 'type' column and transforms 
them into corresponding numerical values, which are then stored in the variable y'''
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['type'])  # Encoding the target variable

'''label_encoder.classes_: This attribute of the LabelEncoder object contains the original categorical classes or labels 
that were encoded into numerical values.

[str(cls) for cls in label_encoder.classes_]:
This is a list comprehension, a compact way to create lists in Python.
It iterates through each class in label_encoder.classes_.
str(cls): Converts each class label to a string format.
The result is a list (class_names) containing the string representations of the original categorical classes that were 
present in the 'type' column before encoding.'''
class_names = [str(cls) for cls in label_encoder.classes_]

'''df.select_dtypes(include=['object']): This DataFrame method, select_dtypes, filters the columns based on their data 
types. The parameter include=['object'] specifies that it should include columns with object data type, which often 
holds categorical data in Pandas DataFrames.

.columns: This attribute retrieves the column index (or labels) of the filtered DataFrame.

.tolist(): Converts the retrieved column index, which is in Pandas Index format, into a Python list.'''
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

'''label_encoders = {}: Initializes an empty dictionary to store LabelEncoder objects for each categorical column.

for col in categorical_cols:: Iterates through each column label present in the categorical_cols list.

label_encoders[col] = LabelEncoder(): Creates a LabelEncoder object for the current categorical column col and stores 
it in the label_encoders dictionary using the column name as the key.

df[col + '_encoded'] = label_encoders[col].fit_transform(df[col]):

df[col + '_encoded']: Creates a new column in the DataFrame df with the name of the original column col, appended with 
'_encoded'. This column will contain the encoded values for the corresponding categorical column.
label_encoders[col].fit_transform(df[col]): Encodes the values of the current categorical column (df[col]) using the 
fit_transform() method of the LabelEncoder object associated with that column (label_encoders[col]). 
It fits the encoder to the unique categories in the column and transforms the categories into numerical values.
This loop processes each categorical column in categorical_cols, creates a LabelEncoder for each, encodes the 
categorical values, and adds a new column with the encoded values to the DataFrame.'''
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col + '_encoded'] = label_encoders[col].fit_transform(df[col])

'''df.drop(categorical_cols, axis=1, inplace=True):
df: Refers to the DataFrame where the operation will be performed.
drop(): This method in Pandas is used to remove specified rows or columns from a DataFrame.
categorical_cols: Contains a list of column names that were previously identified as categorical columns.
axis=1: Specifies that the operation will be applied along the columns (axis=1). This indicates that we want to drop 
columns, not rows.
inplace=True: This parameter, when set to True, modifies the DataFrame df in place and returns None. 
If False or not specified, it returns a new DataFrame with the specified columns dropped.
Effectively, df.drop(categorical_cols, axis=1, inplace=True) removes the original categorical columns 
(identified in categorical_cols) from the DataFrame df. These columns were previously encoded and replaced with new 
columns containing their encoded values. Dropping them allows the DataFrame to retain only the transformed, numerical 
representations of the categorical data for further analysis or model training.'''
df.drop(categorical_cols, axis=1, inplace=True)

'''df.drop(['type'], axis=1):
df: Refers to the original DataFrame containing the dataset.
['type']: Specifies the column(s) to be dropped. In this case, it's dropping the column named 'type'.
axis=1: Indicates that the operation is along columns.
X = df.drop(['type'], axis=1):
Creates a new DataFrame X that holds the features for modeling.
This line drops the column 'type' from the DataFrame df and assigns the resulting DataFrame (without the 'type' column) 
to the variable X.
The resulting DataFrame X contains all the columns from the original df except for the 'type' column, which is now 
removed.
Essentially, X now represents the dataset with the 'type' column removed, allowing it to be used as a feature matrix for 
training machine learning models, as it contains all the features except the one considered as the target variable.'''
X = df.drop(['type'], axis=1)  # Features

'''X contains the features for modeling, and y holds the corresponding target variable.

train_test_split(X, y, test_size=0.2, random_state=42):

X: Represents the feature matrix used for training and testing.
y: Corresponds to the target variable.
test_size=0.2: Specifies the proportion of the dataset allocated for the test set. Here, it's set to 0.2, indicating 20% 
of the data will be used for testing, while 80% will be used for training.
random_state=42: This parameter sets the seed for the random number generator, ensuring reproducibility. 
It ensures that each time you run this code with the same random_state, the data split will be the same.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42):

X_train, X_test: These variables hold the features split into training and testing sets, respectively.
y_train, y_test: These variables contain the corresponding target variable split into training and testing sets, 
respectively.
This line of code executes the train_test_split() function, which divides the dataset into training and testing subsets, 
allowing the machine learning model to be trained on one part of the data (X_train, y_train) and evaluated on another 
unseen part (X_test, y_test).'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''DecisionTreeClassifier: This is a class in Scikit-learn used for decision tree-based classification.

clf = DecisionTreeClassifier(random_state=42):

clf: This is a variable name (can be any valid variable name) assigned to the DecisionTreeClassifier object.
DecisionTreeClassifier(random_state=42): This creates an instance of the DecisionTreeClassifier.
random_state=42: The random_state parameter is set to 42, which is used to control the randomness during the 
construction of the decision tree. It ensures that the results are reproducible, meaning that each time the model is 
trained, it will produce the same results when the same random state is used.
This line of code creates a Decision Tree Classifier object named clf, ready to be trained on the training data to learn 
patterns and relationships within the features and their corresponding target values.'''
clf = DecisionTreeClassifier(random_state=42)

'''clf: This is the instance of the Decision Tree Classifier that was previously initialized.

fit(X_train, y_train): The fit() method in Scikit-learn is used to train a model. Specifically:

X_train: Represents the feature matrix of the training set. It contains a subset of the original dataset's features 
reserved for training.
y_train: Corresponds to the target variable (labels) of the training set. It contains the corresponding subset of the 
original dataset's target values.
When clf.fit(X_train, y_train) is executed:

The Decision Tree Classifier (clf) learns from the provided training data (X_train, y_train).
It analyzes the patterns and relationships between the features (X_train) and their associated target values (y_train) 
to create a decision tree that can be used for making predictions.
After this line of code is executed, the clf object has been trained on the training data and is ready to make 
predictions on new, unseen data.'''
clf.fit(X_train, y_train)

'''clf: Refers to the instance of the Decision Tree Classifier that has been previously trained using the training data.

predict(X_test): The predict() method in Scikit-learn is used to generate predictions based on the trained model.

X_test: Represents the feature matrix of the test set. It contains a subset of the original dataset's features that the 
model has not seen during training.
When clf.predict(X_test) is executed:

The trained Decision Tree Classifier (clf) applies the learned patterns and decision rules to the features in the test 
set (X_test).
It predicts the target variable values (classification labels) for the test set based on the relationships learned 
during the training phase.
After executing this line of code, the variable y_pred contains the predicted values (class labels) generated by the 
Decision Tree Classifier (clf) for the corresponding features in the test set (X_test). These predicted values can be 
compared with the actual labels (y_test) to evaluate the model's performance.'''
y_pred = clf.predict(X_test)

'''Evaluate the accuracy of the classifier'''
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

class_names_str = list(map(str, label_encoder.classes_))


'''export_graphviz(clf, out_file='tree.dot', feature_names=X.columns, class_names=class_names, filled=True):

export_graphviz() is a function in Scikit-learn that exports a trained decision tree as a Graphviz DOT file.
clf: Refers to the trained Decision Tree Classifier.
out_file='tree.dot': Specifies the output file name where the DOT format representation of the decision tree will be 
saved. In this case, it's named 'tree.dot'.
feature_names=X.columns: Provides the feature names to label the nodes in the tree. X.columns contains the names of the 
features used in the model.
class_names=class_names: Specifies the class names for labeling the different classes in the tree. class_names likely 
contains the names of the classes derived from label encoding.
filled=True: This parameter fills the decision tree nodes with colors representing the majority class.
with open("tree.dot") as f: dot_graph = f.read():

This section opens and reads the content of the 'tree.dot' file into the variable dot_graph. The 'tree.dot' file 
contains the DOT format representation of the decision tree generated by export_graphviz().
graphviz.Source(dot_graph):

graphviz is a Python library used to render Graphviz DOT files.
Source(dot_graph): Creates a source object using the DOT format content (dot_graph) and visualizes it as a graph. 
This will display the decision tree graph based on the contents of 'tree.dot'.
Overall, this code segment exports the decision tree as a DOT file, reads its content, and then uses Graphviz to 
visualize and display the decision tree graph in the Jupyter Notebook or Python environment where this code is executed.
'''
export_graphviz(clf, out_file='tree.dot', feature_names=X.columns, class_names=class_names, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)