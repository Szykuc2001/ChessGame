import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

'''Define column names'''
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

'''Read the dataset'''
pima = pd.read_csv("dataset1.csv", header=0, names=col_names)

'''feature_cols: This list contains the names of columns that are considered as features or attributes for the machine 
learning model. These columns will be used to predict the target variable.
pima.head(): This function displays the first few rows (by default, five rows) of the 'pima' DataFrame. It's a way to 
preview the dataset and check its structure or content to ensure it's loaded correctly.'''
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
pima.head()

'''pima[feature_cols]: Selects specific columns from the DataFrame pima based on the list of column names in 
feature_cols. These columns are considered as features or attributes used for prediction in the machine learning model.
X: Stores the selected columns as feature variables for modeling.
pima.label: Selects the 'label' column from the DataFrame pima. This column typically represents the target variable, 
the variable we want to predict.
y: Stores the 'label' column data as the target variable for the machine learning model.
In summary, X contains the selected columns (features) from the dataset, while y holds the target variable values. 
This separation allows for supervised learning where the model learns patterns in the features (X) to predict the 
target variable (y).'''
X = pima[feature_cols] # Features
y = pima.label # Target variable

'''X contains the features for modeling, and y holds the corresponding target variable.

train_test_split(X, y, test_size=0.3, random_state=1):

X: Represents the feature matrix used for training and testing.
y: Corresponds to the target variable.
test_size=0.3: Specifies the proportion of the dataset allocated for the test set. Here, it's set to 0.3, 
indicating 30% of the data will be used for testing, while 70% will be used for training.
random_state=1: This parameter sets the seed for the random number generator, ensuring reproducibility. 
It ensures that each time you run this code with the same random_state, the data split will be the same.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1):

X_train, X_test: These variables hold the features split into training and testing sets, respectively.
y_train, y_test: These variables contain the corresponding target variable split into training and testing sets, 
respectively.
This line of code executes the train_test_split() function, which divides the dataset into training and testing subsets. 
The model will be trained on X_train (features) and y_train (target) and then evaluated on X_test and y_test respectively 
to assess its performance.'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


'''Create Decision Tree classifer object'''
clf = DecisionTreeClassifier()

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
clf = clf.fit(X_train,y_train)

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


'''Evaluate the classifier'''
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


'''dot_data = StringIO(): Initializes an empty string buffer StringIO to store the Graphviz DOT format representation of 
the decision tree.

export_graphviz(): This function from Scikit-learn exports the decision tree as a Graphviz DOT file.

clf: Refers to the trained Decision Tree Classifier.
out_file=dot_data: Specifies that the DOT format representation will be stored in the dot_data buffer.
filled=True: Fills the nodes of the decision tree with colors to represent the majority class.
rounded=True: Makes the decision tree nodes appear with rounded corners.
special_characters=True: Handles special characters in node names or labels.
feature_names=feature_cols: Provides the feature names to label the nodes in the tree.
class_names=['0', '1']: Specifies the class names for labeling the different classes in the tree. Here, '0' and '1' are 
used as class labels.
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()): Reads the DOT format data from dot_data buffer and creates 
a graph object using the pydotplus library.

graph.write_png('diabetes.png'): Writes the graph (decision tree) to a PNG image file named 'diabetes.png'. 
This step saves the decision tree visualization as an image file.

Image(graph.create_png()): Displays the decision tree image using IPython's Image function.'''
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())