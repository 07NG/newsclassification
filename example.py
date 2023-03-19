from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the probabilities of the categories for the testing data
y_pred_prob = model.predict_proba(X_test)

# Evaluate the model using the accuracy metric
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)