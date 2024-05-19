#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Loading all required libraries

# In[2]:


# Loading All Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[3]:


# Loading the datsets
train_data = pd.read_csv('/kaggle/input/competition-dataset/train_kaggle-2.csv')
test_data = pd.read_csv('/kaggle/input/competition-dataset/test_kaggle.csv')


# In[4]:


train_data.head()


# In[5]:


# Check for missing values in each column
missing_values = test_data.isnull().sum()

# Print the count of missing values in each column
missing_values


# In[6]:


test_data.head()


# In[7]:


duplicate_customer_ids = test_data[test_data.duplicated(['Customer_ID'], keep=False)]

if len(duplicate_customer_ids) > 0:
    print("Duplicate Customer_IDs found:")
    print(duplicate_customer_ids[['Customer_ID']])
else:
    print("No duplicate Customer_IDs found.")


# In[8]:


train_data.shape


# In[9]:


test_data.shape


# # Data Preprocessing & Exploratory Data Analysis

# In[10]:


# Check the data types for training data
train_data.dtypes


# In[11]:


# Check the data types for testing data
test_data.dtypes


# When looking at this dataset it was noticed that there are some non-numeric data where its supposed to be numeric. To address this issue, the sales columns were converted to numeric and any numerics were replaced with NaN for initial processing.

# In[12]:


# Replace non-numeric values with NaN for training data
train_data['luxury_sales'] = pd.to_numeric(train_data['luxury_sales'], errors='coerce')
train_data['fresh_sales'] = pd.to_numeric(train_data['fresh_sales'], errors='coerce')
train_data['dry_sales'] = pd.to_numeric(train_data['dry_sales'], errors='coerce')


# In[13]:


# Replace non-numeric values with NaN for testing data
test_data['luxury_sales'] = pd.to_numeric(test_data['luxury_sales'], errors='coerce')
test_data['fresh_sales'] = pd.to_numeric(test_data['fresh_sales'], errors='coerce')
test_data['dry_sales'] = pd.to_numeric(test_data['dry_sales'], errors='coerce')


# Next the datatypes were converted

# In[14]:


# Convert Data Types Accordingly
# Convert Customer_ID to string
train_data['Customer_ID'] = train_data['Customer_ID'].astype(str)
test_data['Customer_ID'] = test_data['Customer_ID'].astype(str)

# Convert luxury_sales, fresh_sales, and dry_sales to float
train_data['luxury_sales'] = train_data['luxury_sales'].astype(float)
train_data['fresh_sales'] = train_data['fresh_sales'].astype(float)
train_data['dry_sales'] = train_data['dry_sales'].astype(float)

# Convert luxury_sales, fresh_sales, and dry_sales to float
test_data['luxury_sales'] = test_data['luxury_sales'].astype(float)
test_data['fresh_sales'] = test_data['fresh_sales'].astype(float)
test_data['dry_sales'] = test_data['dry_sales'].astype(float)


# In[15]:


# Check the data types
train_data.dtypes


# In[16]:


test_data.dtypes


# In[17]:


train_data.describe()


# The datatypes look okay now. So let's handle the missing values and duplicates if any.

# ## Handling Missing Values

# In[18]:


# Check for missing values in each column
missing_values = train_data.isnull().sum()

# Print the count of missing values in each column
missing_values


# In[19]:


# Check for missing values in each column
missing_values = test_data.isnull().sum()

# Print the count of missing values in each column
missing_values


# In[20]:


# Imputing Values

# Impute missing values with mean
train_data['luxury_sales'] = train_data['luxury_sales'].fillna(train_data['luxury_sales'].mean())
train_data['fresh_sales'] = train_data['fresh_sales'].fillna(train_data['fresh_sales'].mean())
train_data['dry_sales'] = train_data['dry_sales'].fillna(train_data['dry_sales'].mean())


# In[21]:


# Impute missing values with mean
test_data['luxury_sales'] = test_data['luxury_sales'].fillna(test_data['luxury_sales'].mean())
test_data['fresh_sales'] = test_data['fresh_sales'].fillna(test_data['fresh_sales'].mean())
test_data['dry_sales'] = test_data['dry_sales'].fillna(test_data['dry_sales'].mean())


# In[22]:


# Find the most frequent city in the 'outlet_city' column
most_frequent_city_train = train_data['outlet_city'].mode()[0]
most_frequent_city_train


# In[23]:


# Find the most frequent city in the 'outlet_city' column
most_frequent_city_test = test_data['outlet_city'].mode()[0]
most_frequent_city_test


# In[24]:


# Count the number of unique outlets
num_unique_outlets = train_data['outlet_city'].nunique()

# Print the unique outlets
unique_outlets = train_data['outlet_city'].unique()
print("Number of unique outlets:", num_unique_outlets)
print("Unique outlets:", unique_outlets)


# In[25]:


# Count the number of unique outlets
num_unique_outlets = test_data['outlet_city'].nunique()

# Print the unique outlets
unique_outlets = test_data['outlet_city'].unique()
print("Number of unique outlets:", num_unique_outlets)
print("Unique outlets:", unique_outlets)


# In[26]:


# Impute the missing values in the 'outlet_city' column with the most frequent city
train_data['outlet_city'] = train_data['outlet_city'].fillna(most_frequent_city_train)
test_data['outlet_city'] = test_data['outlet_city'].fillna(most_frequent_city_test)


# In[27]:


# Replace incorrect outlet names with correct ones
train_data['outlet_city'] = train_data['outlet_city'].replace({'MoraTuwa': 'Moratuwa',
                                                    'PeliyagodA': 'Peliyagoda',
                                                    'batticaloa': 'Batticaloa',
                                                    'Trincomale': 'Trincomalee',
                                                    'kalmunai': 'Kalmunai'})

# Count the number of unique outlets after corrections
num_unique_outlets_corrected = train_data['outlet_city'].nunique()

# Print the unique outlets after corrections
unique_outlets_corrected = train_data['outlet_city'].unique()
print("Number of unique outlets after corrections:", num_unique_outlets_corrected)
print("Unique outlets after corrections:", unique_outlets_corrected)


# In[28]:


# Replace incorrect outlet names with correct ones
test_data['outlet_city'] = test_data['outlet_city'].replace({'MoraTuwa': 'Moratuwa',
                                                    'PeliyagodA': 'Peliyagoda',
                                                    'batticaloa': 'Batticaloa',
                                                    'Trincomale': 'Trincomalee',
                                                    'kalmunai': 'Kalmunai'})

# Count the number of unique outlets after corrections
num_unique_outlets_corrected = test_data['outlet_city'].nunique()

# Print the unique outlets after corrections
unique_outlets_corrected = test_data['outlet_city'].unique()
print("Number of unique outlets after corrections:", num_unique_outlets_corrected)
print("Unique outlets after corrections:", unique_outlets_corrected)


# In[29]:


train_data.head()


# In[30]:


test_data.head()


# In[31]:


# Remove the '.0' suffix from the 'Customer_ID' column
train_data['Customer_ID'] = train_data['Customer_ID'].str.rstrip('.0')


# The sales values were imputed using mean. The outlet city was imputed using the most frequent city. And as mentioned in the case study we can see that there are only 22 outlets.

# In[32]:


train_data.shape


# In[33]:


test_data.shape


# # Handle null values and erroneous values in cluster category

# In[34]:


duplicate_customer_ids = test_data[test_data.duplicated(['Customer_ID'], keep=False)]

if len(duplicate_customer_ids) > 0:
    print("Duplicate Customer_IDs found:")
    print(duplicate_customer_ids[['Customer_ID']])
else:
    print("No duplicate Customer_IDs found.")


# In[35]:


# Find missing values in train_data
missing_values = train_data.isnull().sum()

# Print the count of missing values in each column
missing_values


# In[36]:


# Drop this null cluster category
train_data.dropna(subset=['cluster_catgeory'], inplace=True)


# In[37]:


# Find unique values in the 'cluster_catgeory' column
unique_cluster_categories = train_data['cluster_catgeory'].unique()

# Print unique values
unique_cluster_categories


# In[38]:


# Remove values greater than 6 and correct inconsistent representations
train_data['cluster_catgeory'] = train_data['cluster_catgeory'].replace({'6\\': '6', 1: '1', 5: '5', 4: '4', 2: '2', 3: '3', 6:'6'})

# Keep only values up to 6
train_data = train_data[train_data['cluster_catgeory'].isin(['1', '2', '3', '4', '5', '6'])]

# Find unique values in the 'cluster_catgeory' column again
unique_cluster_categories = train_data['cluster_catgeory'].unique()

# Print unique values
print(unique_cluster_categories)


# There should only be 06 cluster categories (1,2,3,4,5,and 6).

# In[39]:


train_data.shape


# In[40]:


# Find missing values in train_data
missing_values = train_data.isnull().sum()

# Print the count of missing values in each column
missing_values


# Since we need an accurate classification of the cluster category for our training data we can opt to remove any missing values in the training dataset pertaining to cluster category

# In[41]:


train_data.shape


# In[42]:


test_data.shape


# In[43]:


train_data.describe()


# In[44]:


test_data.describe()


# # Feature Engineering & Data Exploration

# In[45]:


# Calculate total sales
train_data['total_sales'] = train_data['luxury_sales'] + train_data['fresh_sales'] + train_data['dry_sales']

# Calculate the percentage of each type of sale
luxury_percent = (train_data['luxury_sales'].sum() / train_data['total_sales'].sum()) * 100
fresh_percent = (train_data['fresh_sales'].sum() / train_data['total_sales'].sum()) * 100
dry_percent = (train_data['dry_sales'].sum() / train_data['total_sales'].sum()) * 100

# Create labels and sizes for the pie chart
labels = ['Luxury Sales', 'Fresh Sales', 'Dry Sales']
sizes = [luxury_percent, fresh_percent, dry_percent]

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#97D8ED', '#F0E593', '#F8BC43'])
plt.title('Percentage of Luxury, Fresh, and Dry Sales of Total Sales')
plt.axis('equal')
plt.show()


# In[46]:


# Plot the count of transactions for each outlet_city
plt.figure(figsize=(12, 6))
sns.countplot(data=train_data, x='outlet_city', order=train_data['outlet_city'].value_counts().index)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Count of Transactions for Each Outlet City')
plt.xlabel('Outlet City')
plt.ylabel('Count')
plt.show()


# In[47]:


plt.figure(figsize=(12, 6))

# Use seaborn's built-in pastel color palette
sns.countplot(
    data=train_data, 
    x='outlet_city', 
    order=train_data['outlet_city'].value_counts().index,
    palette='pastel'
)

plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Count of Transactions for Each Outlet City')
plt.xlabel('Outlet City')
plt.ylabel('Count')
plt.show()


# In[48]:


# Compute the total sales for each city
city_total_sales = train_data.groupby('outlet_city').agg({'luxury_sales': 'sum', 'fresh_sales': 'sum', 'dry_sales': 'sum'}).sum(axis=1)

# Plot the total sales amount for each outlet city
plt.figure(figsize=(12, 6))
sns.barplot(x=city_total_sales.index, y=city_total_sales.values)
plt.xticks(rotation=90)
plt.title('Total Sales Amount for Each Outlet City')
plt.xlabel('Outlet City')
plt.ylabel('Total Sales Amount')
plt.show()


# In[49]:


# Compute the total sales for each city
city_total_sales = train_data.groupby('outlet_city').agg({'luxury_sales': 'sum', 'fresh_sales': 'sum', 'dry_sales': 'sum'}).sum(axis=1)

plt.figure(figsize=(12, 6))

# Use seaborn's built-in pastel color palette
sns.barplot(x=city_total_sales.index, y=city_total_sales.values, palette='pastel')

plt.xticks(rotation=90)
plt.title('Total Sales Amount for Each Outlet City')
plt.xlabel('Outlet City')
plt.ylabel('Total Sales Amount')
plt.show()


# In[50]:


# Find the city with the highest total sales
city_highest_sales = city_total_sales.idxmax()
highest_sales_amount = city_total_sales.max()

# Find the city with the lowest total sales
city_lowest_sales = city_total_sales.idxmin()
lowest_sales_amount = city_total_sales.min()

print("City with the highest total sales:", city_highest_sales)
print("Highest total sales amount:", highest_sales_amount)
print("\nCity with the lowest total sales:", city_lowest_sales)
print("Lowest total sales amount:", lowest_sales_amount)


# In[51]:


# Get unique cities
cities = train_data['outlet_city'].unique()

# Set up the figure and axes
plt.figure(figsize=(12, 6))

# Iterate over each city and plot the breakdown of sales
for city in cities:
    city_data = train_data[train_data['outlet_city'] == city]
    total_luxury_sales = city_data['luxury_sales'].sum()
    total_fresh_sales = city_data['fresh_sales'].sum()
    total_dry_sales = city_data['dry_sales'].sum()
    
    # Plot the breakdown for each city
    plt.bar(city, total_luxury_sales, color='#A6620A', label='Luxury Sales' if city == cities[0] else None)
    plt.bar(city, total_fresh_sales, color='#F8BC43', bottom=total_luxury_sales, label='Fresh Sales' if city == cities[0] else None)
    plt.bar(city, total_dry_sales, color='#F0E593', bottom=total_luxury_sales + total_fresh_sales, label='Dry Sales' if city == cities[0] else None)

# Add labels and title
plt.title('Breakdown of Sales for Each City')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend()  # Add legend
plt.tight_layout()
plt.show()


# # Cluster Category Exploration

# In[52]:


train_data.head()


# In[53]:


# Find the count of transactions for each cluster
transaction_count_by_cluster = train_data['cluster_catgeory'].value_counts()

# Print the count of transactions for each cluster
print(transaction_count_by_cluster)


# In[54]:


# Calculate total sales for each cluster
total_sales_by_cluster = train_data.groupby('cluster_catgeory')[['luxury_sales', 'fresh_sales', 'dry_sales']].sum()

# Print the total sales for each cluster
total_sales_by_cluster


# In[55]:


# Find the category with the highest total sales
highest_total_sales_category = total_sales_by_cluster.sum(axis=1).idxmax()

# Find the category with the highest luxury sales
highest_luxury_sales_category = total_sales_by_cluster['luxury_sales'].idxmax()

# Find the category with the highest fresh sales
highest_fresh_sales_category = total_sales_by_cluster['fresh_sales'].idxmax()

# Find the category with the highest dry sales
highest_dry_sales_category = total_sales_by_cluster['dry_sales'].idxmax()

# Print the results
print("Category with the highest Total Sales:", highest_total_sales_category)
print("Category with the highest Luxury Sales:", highest_luxury_sales_category)
print("Category with the highest Fresh Sales:", highest_fresh_sales_category)
print("Category with the highest Dry Sales:", highest_dry_sales_category)


# In[56]:


# Group the data by cluster category and find the mode of outlet city in each group
most_common_city_by_category = train_data.groupby('cluster_catgeory')['outlet_city'].agg(lambda x: x.mode()[0])

# Print the result
print("Most common city in each category:")
print(most_common_city_by_category)


# In[57]:


# Group the data by cluster category and calculate minimum and maximum sales for each sale type
min_max_sales_by_category = train_data.groupby('cluster_catgeory').agg({
    'luxury_sales': ['min', 'max'],
    'fresh_sales': ['min', 'max'],
    'dry_sales': ['min', 'max']
})

# Print the result
print("Minimum and maximum sales in each category for each sale type:")
min_max_sales_by_category


# In[58]:


# Group the data by outlet city and sum the sales for each city
total_sales_by_city = train_data.groupby('outlet_city')[['luxury_sales', 'fresh_sales', 'dry_sales']].sum()

# Calculate the total sales for each city by adding luxury, fresh, and dry sales
total_sales_by_city['total_sales'] = total_sales_by_city.sum(axis=1)

# Sort the cities by total sales in descending order and get the top 5
top_5_cities = total_sales_by_city.sort_values(by='total_sales', ascending=False).head(5)

# Print the top 5 cities with the highest total sales
print("Top 5 Outlet Cities with the Highest Total Sales:")
top_5_cities


# In[59]:


test_data['total_sales'] = test_data['luxury_sales'] + test_data['fresh_sales'] + test_data['dry_sales']


# # Feature Engineering

# ## Categorical Encoding

# In[60]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert all values to strings
train_data['outlet_city'] = train_data['outlet_city'].astype(str)

# Fit and transform the target variable
train_data['outlet_city_encoded'] = label_encoder.fit_transform(train_data['outlet_city'])

# Create a dictionary to map encoded labels to original outlet names
label_map = {label: city for label, city in zip(label_encoder.transform(train_data['outlet_city']), train_data['outlet_city'])}

# Print the mapping
print("Encoded Label -> Outlet Name:")
for label, city in label_map.items():
    print(f"{label} -> {city}")


# In[61]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Concatenate train and test data for encoding consistency
combined_data = pd.concat([train_data, test_data])

# Convert all values to strings
combined_data['outlet_city'] = combined_data['outlet_city'].astype(str)

# Fit and transform the target variable
combined_data['outlet_city_encoded'] = label_encoder.fit_transform(combined_data['outlet_city'])

# Create a dictionary to map encoded labels to original outlet names
label_map = {label: city for label, city in zip(label_encoder.transform(combined_data['outlet_city']), combined_data['outlet_city'])}

# Print the mapping
print("Encoded Label -> Outlet Name:")
for label, city in label_map.items():
    print(f"{label} -> {city}")
# Splitting back into train and test sets based on index range
train_data = combined_data.iloc[:len(train_data)]
test_data = combined_data.iloc[len(train_data):]


# Verify encoded values in test set
print(test_data[['outlet_city', 'outlet_city_encoded']].head())


# In[62]:


duplicate_customer_ids = test_data[test_data.duplicated(['Customer_ID'], keep=False)]

if len(duplicate_customer_ids) > 0:
    print("Duplicate Customer_IDs found:")
    print(duplicate_customer_ids[['Customer_ID']])
else:
    print("No duplicate Customer_IDs found.")


# In[63]:


test_data.shape


# In[64]:


test_data.head()


# In[65]:


# Drop the cluster_category column from test_data
test_data = test_data.drop(columns=['cluster_catgeory'])

# Verify that the column has been dropped
test_data.head()


# ## Feature Scaling of Sales Values

# In[66]:


# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical features in the training data
train_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales']] = scaler.fit_transform(train_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales']])
test_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales']] = scaler.fit_transform(test_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales']])


# In[67]:


test_data.head()


# In[68]:


test_data.shape


# In[69]:


duplicate_customer_ids = test_data[test_data.duplicated(['Customer_ID'], keep=False)]

if len(duplicate_customer_ids) > 0:
    print("Duplicate Customer_IDs found:")
    print(duplicate_customer_ids[['Customer_ID']])
else:
    print("No duplicate Customer_IDs found.")


# ## Encoding the Cluster Category Column

# In[70]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'cluster_catgeory' column in the training data
train_data['cluster_catgeory_encoded'] = label_encoder.fit_transform(train_data['cluster_catgeory'].astype(str))+1

# Display the unique encoded values
print("Encoded Unique Values for cluster_catgeory:", train_data['cluster_catgeory_encoded'].unique())


# In[71]:


train_data.head()


# # Model Building

# In[72]:


# Selecting columns for the modeling dataset
training_data = train_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales', 'outlet_city_encoded', 'cluster_catgeory_encoded']]

# Display the new DataFrame
training_data.head()


# In[73]:


# Assuming your dataframe is named training_data
training_data_copy = training_data.copy()
training_data_copy.rename(columns={'cluster_catgeory_encoded': 'cluster_category', 'outlet_city_encoded': 'outlet_city'}, inplace=True)
training_data = training_data_copy


# In[74]:


# Compute the correlation matrix
correlation_matrix = training_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()


# In[97]:


# Compute the correlation matrix
correlation_matrix = training_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))

# Use seaborn's pastel color palette for the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='Pastel1', fmt=".2f", linewidths=0.5)

plt.title('Feature Correlation Heatmap')
plt.show()


# # Spliting Train and Test Set

# In[75]:


# Assuming your dataframe is named training_data
training_dataa = training_data.sample(n=400000, random_state=42)  # Randomly sample 400,000 data points


# In[76]:


# Define the features and target variable
X = training_dataa.drop(columns=['cluster_category'])  
y = training_dataa['cluster_category'] 


# In[77]:


X.head()


# In[78]:


y.head()


# In[79]:


# Split the data into train and test sets (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[80]:


X_train.shape


# In[81]:


X_test.shape


# In[82]:


test_data.head()


# # Modeling using Random Forest

# In[83]:


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred = rf_classifier.predict(X_test)


# # Check model accuracy

# In[84]:


# Evaluate the accuracy of the model Random Forest
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[85]:


duplicate_customer_ids = test_data[test_data.duplicated(['Customer_ID'], keep=False)]

if len(duplicate_customer_ids) > 0:
    print("Duplicate Customer_IDs found:")
    print(duplicate_customer_ids[['Customer_ID']])
else:
    print("No duplicate Customer_IDs found.")


# In[86]:


# Standardize the features by scaling them to have mean 0 and variance 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the KNN classifier on the training data
knn.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test_scaled)


# In[87]:


# Evaluate the accuracy of the KNN model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[88]:


test_data.head()


# In[89]:


train_data.head()


# In[90]:


train_data_new = train_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales', 'outlet_city_encoded', 'cluster_catgeory_encoded']]


# In[91]:


train_data_new.head()


# In[92]:


test_data_new = test_data[['luxury_sales', 'fresh_sales', 'dry_sales', 'total_sales', 'outlet_city_encoded']]


# In[93]:


test_data_new.head()


# In[94]:


# Assuming train_data and test_data are DataFrames with the same structure
# Prepare the data
X_train = train_data_new.drop(columns=['cluster_catgeory_encoded'])  # Features for training
y_train = train_data_new['cluster_catgeory_encoded']  # Target variable for training

X_test = test_data_new # Features for testing

# Initialize KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the model
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)


# In[95]:


predictions_df = pd.DataFrame({
    'Customer_ID': test_data['Customer_ID'],  # Customer_ID from train_data
    'cluster_category': y_pred  # Predicted cluster_category
})

# Save predictions to CSV
predictions = predictions_df.to_csv('predictions.csv', index=False)

# Verify that the CSV file is saved
print("Predictions saved to predictions.csv")


# In[96]:


predictions_df

