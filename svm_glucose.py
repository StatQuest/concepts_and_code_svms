# first we import the modules we're going to use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# now we open the data file and print out the
# first few lines...
df = pd.read_csv("data_diabetes.csv")
df.head()

# now plot both features and see how well
# they separate the people with diabetes
sns.scatterplot(df, 
        x="bloodpressure", 
        y="glucose",
        style=df["diabetes"],
        hue=df['diabetes'],
        s=150,
        alpha=0.5,
        palette=["orange", "blue"])

# now let's see how well
# glucose can separate the people with
# diabetes
sns.scatterplot(df, 
        x="glucose", 
        y=0,
        style=df["diabetes"],
        hue=df['diabetes'],
        s=150,
        alpha=0.5,
        palette=["orange", "blue"])

# now let's check for missing data.
# first just make sure there are no weird characters
# in the data (we're just expecting numbers)
# we can do this by looking at the datatypes for each 
# column. They should a be some type of numeric data.
print(df.dtypes)
# Now count up the number of null values
df.isnull().sum()
# none! So we're good to go there.

# Now we'll use "glucose" to predict "diabetes".
# To make our lives a little easier, we'll just remove the
# "glucose" column.

# Now split the data into independent variables (features), X
# and a dependent variable (target), y
X = df.drop(['diabetes', 'bloodpressure'], axis=1)
X.head()

# NOTE: The suport vector classfier function, CVC(),
#       expects the dependent variable
#       to be a 1d array, not a dataframe. This conversion
#       happens by default when we save just a single column.
#       That said, if we wanted to retain the pandas dataframe
#       stucture for just a single column we could do it by adding
#       an additional pair of square brackets, "[]", around the column
#       name like this: y = df[['diabetes']]. In this case, y would
#       continue to be a dataframe.
y = df['diabetes'] 
y.head()

# now split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# now let's standardize the data
# this means we need to scale the features so that 
# they have a mean of 0 and a standard deviation of 1
# And we do this by subtracting the mean from each value
# and dividing by the standard deviation. However, to avoid
# data leakage, we calculate the mean and standard deviation from
# just the training data and use those values to scale both
# the training and testing data. We'll do this by using
# a StandardScaler().
scaler = StandardScaler().set_output(transform="pandas")
# Now calculate the mean and standard deviation from the training
# data and standardize it with fit_transform()
X_train_scaled = scaler.fit_transform(X_train)
# Now standardize the testingn data using the same mean and 
# standard deviation with transform()
X_test_scaled = scaler.transform(X_test)
X_test_scaled.head()

# now, using just the testing data,
# let's just see how well glucose alone
# can predict diabetes
sns.scatterplot(X_test_scaled, 
        x="glucose", 
        y=0,
        style=y_test,
        hue=y_test,
        s=150,
        alpha=0.5,
        palette=["orange", "blue"])
# and we see some overlap still. In other words, the testing
# dataset still needs a support vector classifier for classification.

# now let's build a preliminary SVM model using the polynomial kernel
# in this example, we're using the parameters that I demonstrated in
# the statquest video: Support Vector Machines Part 2: The Polynomial Kernel
poly_model = SVC(kernel='poly', degree=2, coef0=0.5)
poly_model.fit(X_train_scaled, y_train)

# now let's run the test data through the model
y_pred = poly_model.predict(X_test_scaled)
# and draw a confusion matrix
poly_confusion_matrix = pd.crosstab(y_test, 
                                y_pred, 
                                rownames=['Actual'], 
                                colnames=['Predicted'])
print(poly_confusion_matrix)
# now let's calculate the accuracy score
poly_accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {poly_accuracy:.2f}")
# Bam. The support vector classifer does a pretty good job
# classifying who has diabetes with just the glucose data.

# Now that we can use the polynomial kernel
# to create a classifier, it would be nice if we could
# plot the transformed data (with the extra dimension added
# to it) and the Support Vector Classifer that we just created.
# However, there is no way to extract the x/y coordinates for the
# data from the current classifer, this is because the data
# are not actually transformed before classification, but use the 
# "kernal trick" to calculate the distances from each point and the
# classifier.
# So, if we want to visualize things, we have to do everything by
# hand and manually transform the data into the higher dimensions.
# So we'll  do the transformation by hand and then use the "linear"
# kernel find a straight line for classification.

# So, first, lets make a copy of the training data...
X_linear_train = X_train_scaled.copy()
# ...and then create a new dimension with glucose squared.
X_linear_train["glucose^2"] = X_train_scaled ** 2
X_linear_train.head()

# Now do the same thing with the testing data.
X_linear_test = X_test_scaled.copy()
X_linear_test["glucose^2"] = X_test_scaled ** 2
X_linear_test.head()

# now create a support vector classifier, but
# because the data have already been transformed into the
# higher dimensions, we can use the "linear" kernel to just
# find a linear classifier without doing a transformation.
linear_model = SVC(kernel='linear')
linear_model.fit(X_linear_train, y_train)

# now run the manually transformed test data through new model
y_pred_linear = linear_model.predict(X_linear_test)

# now create a confusion matrix for the linear kernel
linear_confusion_matrix = pd.crosstab(y_test, 
                                y_pred_linear, 
                                rownames=['Actual'], 
                                colnames=['Predicted'])
print(linear_confusion_matrix)
# and compare to the confusion matrix with the polynomial kernel
print(poly_confusion_matrix)
# and we see that both models give us the same confusion matrix.
# BAM!

# now calculate the accuracy for the linear kernel
linear_accuracy = accuracy_score(y_pred_linear, y_test)
print(f"Linear accuracy: {linear_accuracy:.2f}")
# and compare to the accuracy of the polynomial kernel
print(f"Polynomial accuracy: {poly_accuracy:.2f}")
# and we see that both models give us the same accuracy
# DOULBE BAM!!

# Now let's draw the testing data and the classifier

# The first thing we need to do  is calculate the
# slope and intercept for the linear classifier.
# NOTE: These coefficients calculated by the model
# are not the slope and intercept (which would be nice) 
# instead, we get w0, w1 and b in the following equation:
# w0*x + w1*y + b = 0
w = linear_model.coef_[0]
b = linear_model.intercept_[0]

# Now transform the coefficients into the slope and intercept
# For a 2D line: w0*x + w1*y + b = 0  =>  y = (-w0/w1)*x + (-b/w1)
slope = -w[0] / w[1]
print(slope)
y_intercept = -b / w[1]
print(y_intercept)

# now plot the transformed test data and the classifier
fig, ax = plt.subplots()
sns.scatterplot(data=X_linear_test, 
        x="glucose", 
        y="glucose^2",
        style=y_test,
        hue=y_test,         
        s=150,
        alpha=0.4,
        palette=["orange", "blue"],
        ax=ax)
ax.axline((0, y_intercept), 
            slope=slope, 
            color='red', 
            label='classifier')
plt.show()
# and now we see how the classifier works. TRIPLE BAM!!!
