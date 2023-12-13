# %% [markdown]
# # Job Attrition Prediction Model
# 
# Data source:
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# %% [markdown]
# About Dataset
# Uncover the factors that lead to employee attrition and explore important questions such as ‘show me a breakdown of distance from home by job role and attrition’ or ‘compare average monthly income by education and attrition’. This is a fictional data set created by IBM data scientists.
# 
# Education
# 1 'Below College'
# 2 'College'
# 3 'Bachelor'
# 4 'Master'
# 5 'Doctor'
# 
# EnvironmentSatisfaction
# 1 'Low'
# 2 'Medium'
# 3 'High'
# 4 'Very High'
# 
# JobInvolvement
# 1 'Low'
# 2 'Medium'
# 3 'High'
# 4 'Very High'
# 
# JobSatisfaction
# 1 'Low'
# 2 'Medium'
# 3 'High'
# 4 'Very High'
# 
# PerformanceRating
# 1 'Low'
# 2 'Good'
# 3 'Excellent'
# 4 'Outstanding'
# 
# RelationshipSatisfaction
# 1 'Low'
# 2 'Medium'
# 3 'High'
# 4 'Very High'
# 
# WorkLifeBalance
# 1 'Bad'
# 2 'Good'
# 3 'Better'
# 4 'Best'

# %% [markdown]
# ## Importing required libraries

# %% [markdown]
#  

# %%
import numpy as np 
import pandas as pd

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# %% [markdown]
# > Ignoring any kind of unwanted warings in while analysing dataset.
# 

# %%
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Data Analysis

# %% [markdown]
# ### Studying Data Labels 

# %%
jAttr_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
jAttr_data.info()

# %%
jAttr_data.describe().round(2).T

# %% [markdown]
# ### Checking and removing unnecessary features

# %%
jAttr_data['EmployeeCount'].value_counts()

# %% [markdown]
# ***- From above observations it can be concluded that the EmployeeCount column has only one value and hence is a insignificant column for prediction purpose and needs to be dropped.***

# %%
jAttr_data['Over18'].value_counts()

# %% [markdown]
# ***- A similar thing can be observed in the column of Over18. Which indicates all the employees are Over 18 years of age and hence the column need not to be considered for model training purpose.***

# %% [markdown]
# ***- Employee Number has nothing to do with the Attrition hence it can be safely dropped.***

# %%
jAttr_data['StandardHours'].value_counts()

# %% [markdown]
# ***- This value can also be droped as entire column has same value***

# %%
fig,axes= plt.subplots(nrows=1, ncols= 4, figsize=(16,5))
fig.subplots_adjust(wspace=0.4)
axes[0].pie(jAttr_data['EmployeeCount'].value_counts())



# %%
jAttr_data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1,inplace=True)
jAttr_data.dtypes.sort_values()

# %% [markdown]
# ### Checking the distribution of Target variable-'Attrition' 

# %%
attrition_counts = jAttr_data['Attrition'].value_counts()
labels = ['Retention', 'Attrition']
# plt.pie(attrition_counts, labels=labels, autopct='%1.2f%%',colors=['blue','red'])
plt.pie(attrition_counts, labels=labels, autopct='%1.2f%%', colors=['#FFC300', '#900C3F'])
plt.title('Retention vs Attrition Distribution')
plt.show()

# %% [markdown]
# > 16.12% employees have left the company.

# %%
plt.figure(figsize = (5, 5))
sns.histplot(x='Age', hue='Attrition', data=jAttr_data, kde=True, palette='magma')


# %% [markdown]
# > Major Attrition can be seen in the age group of 26-38.<br/>Also it can be noted that most of the employees are below age of 50. 

# %% [markdown]
# ### Analyzing other aspects of Dataset

# %%
gender_counts = jAttr_data['Gender'].value_counts()
labels = ['Male', 'Female']
fig, axes =plt.subplots(nrows=1, ncols=2, figsize=(7,4))
fig.subplots_adjust(wspace=0.6)
axes[0].pie(gender_counts, labels=labels, autopct='%1.2f%%',colors=['blue','red'])
axes[0].set_title('Gender Distribution of the company')

# Create a pivot table with counts of Attrition by Gender
pivot_table = jAttr_data.pivot_table(index='Gender', columns='Attrition', aggfunc='size')

# Calculate the percentage of attrition in male and female
total_by_gender = pivot_table.sum(axis=1)
attrition_percent = pivot_table['Yes'] / total_by_gender * 100

# Create a stacked bar chart
ax = sns.barplot(x=pivot_table.index, y=pivot_table.values[:, 0], color='#F9E79F', label='No')
ax = sns.barplot(x=pivot_table.index, y=pivot_table.values[:, 1], color='#900C3F', label='Yes')

# Add text annotations with the percentage of attrition
for i, v in enumerate(attrition_percent):
    ax.text(i, pivot_table.values[i, 1]/2, f'{v:.1f}%', ha='center', va='center', fontsize=11)

# Add legend, title, and axis labels to the chart
plt.legend(title='Attrition')
plt.title('Employee Attrition by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

plt.show()

# %% [markdown]
# > Company has more male employees than female and attrition rate in males are slightly greater.

# %%
plt.figure(figsize = (8 , 8))
plt.subplot(2 ,1,1)
sns.countplot(x= 'DistanceFromHome' ,data =jAttr_data ,palette='YlOrRd')
plt.title('Distance From Home')
plt.subplot(2,1,2)
sns.countplot(x= 'DistanceFromHome' ,data = jAttr_data ,palette='winter_r'  ,hue =jAttr_data['Attrition'])

# %%
# Create bins based on distance ranges
bins = [0, 5, 10, 15, 20, 25, 30]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30']
jAttr_data['DistanceRange'] = pd.cut(jAttr_data['DistanceFromHome'], bins=bins, labels=labels)

plt.figure(figsize = (6 , 4))

# fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(8,8))

sns.countplot(data=jAttr_data,x='DistanceRange',palette='YlOrBr')
# axes[0].set_title('Distance From Home categorial')

# Create a histogram with distance ranges and hue by Attrition
sns.countplot(data=jAttr_data, x='DistanceRange', hue='Attrition', palette='winter_r')

# Add a title and axis labels
plt.title('Employee Attrition by Distance Range')
plt.xlabel('Distance Range')
plt.ylabel('Count')

# Display the chart
plt.show()

jAttr_data.drop(['DistanceRange'],axis =1, inplace=True)

# %%
marital_status_count = jAttr_data['MaritalStatus'].value_counts()
labels=['Married','Single','Divorce']
fig, axes =plt.subplots(nrows=1, ncols=2, figsize=(9,4))
fig.subplots_adjust(wspace=0.6)

axes[0].pie(marital_status_count, labels=labels, autopct='%1.2f%%')
axes[0].set_title('Maritial Status')


# Create a pivot table with counts of Attrition by Gender
pivot_table = jAttr_data.pivot_table(index='MaritalStatus', columns='Attrition', aggfunc='size')

# Calculate the percentage of attrition in male and female
total_by_MaritalStatus = pivot_table.sum(axis=1)
attrition_percent = pivot_table['Yes'] / total_by_MaritalStatus * 100

# Create a stacked bar chart
ax = sns.barplot(x=pivot_table.index, y=pivot_table.values[:, 0], color='blue',  label='No')
ax = sns.barplot(x=pivot_table.index, y=pivot_table.values[:, 1], color= 'red' ,label='Yes')

# Add text annotations with the percentage of attrition
for i, v in enumerate(attrition_percent):
    ax.text(i, pivot_table.values[i, 1]/2, f'{v:.1f}%', ha='center', va='center', fontsize=11)

# Add legend, title, and axis labels to the chart
plt.legend(title='Attrition')
plt.title('Employee Attrition by Marital Status')
plt.xlabel('MaritalStatus')
plt.ylabel('Count')

plt.show()

# %% [markdown]
# > In singles attrition % is maximum

# %%
sns.countplot(data=jAttr_data, x="Department", hue="Attrition", palette='winter_r')
plt.title('Department wise')
jAttr_data['Department'].value_counts()

# %%
sns.countplot(y= 'JobRole' ,data = jAttr_data ,palette='winter_r'  ,hue ='Attrition')
plt.title('Job role wise')


# %% [markdown]
# > Based on the countplot with percentage labels, it appears that the job roles of 'sales executive', 'sales representative', and 'lab technician' have higher proportions of employees leaving the company, compared to other job roles. This suggests that the company may want to focus on retaining employees in these particular roles.

# %%
sns.countplot(data=jAttr_data, x="OverTime", hue="Attrition")

# %% [markdown]
# > No significant difference observed

# %%
data_left=jAttr_data[jAttr_data[ 'Attrition']=='Yes']['YearsWithCurrManager'] 
data_stay=jAttr_data[jAttr_data[ 'Attrition']=='No']['YearsWithCurrManager'] 

sns.kdeplot(data_left, label = 'Employee left', fill=True, color = 'b' )
sns.kdeplot(data_stay, label = 'Employee stay', fill=True, color = 'r')
plt.legend()
plt.show()


# %%
df=jAttr_data.copy()
df.head(5)

# %%
df['Education'].replace([1,2,3,4,5],["Below College","College"," Bachelor","Master","Doctor"],inplace=True)
df['EnvironmentSatisfaction'].replace([1,2,3,4,],["Low ","Medium"," High","Very High"],inplace=True)
df['JobInvolvement'].replace([1,2,3,4],["Low ","Medium"," High","Very High"],inplace=True)
df['JobSatisfaction'].replace([1,2,3,4],["Low ","Medium"," High","Very High"],inplace=True)
df['PerformanceRating'].replace([1,2,3,4],["Low ","Good"," Excellent"," Outstanding"],inplace=True)
df['RelationshipSatisfaction'].replace([1,2,3,4],["Low ","Medium"," High","Very High"],inplace=True)
df['WorkLifeBalance'].replace([1,2,3,4,],["Bad ","Good"," Better"," Best"],inplace=True)
df.head(5)

# %%
labels=["Below College","College","Bachelor","Master","Doctor"]
for i in range(len(labels)):
    print(f"{i+1}: {labels[i]}")
sns.countplot(x= 'Education' ,data =df ,palette='winter_r',hue='Attrition')
plt.title('Education Vs Attrition')
plt.show()

# %% [markdown]
# >Employees who hold a Bachelor's degree are more likely to leave the company compared to those with other degrees.

# %%
plt.figure(figsize=(10,5))
sns.countplot(x= 'EducationField' ,data =df ,palette='winter_r',hue='Attrition')
plt.title('EducationField Vs Attrition')
plt.show()

# %% [markdown]
# > Attrition is high in Life Sciences

# %%
sns.countplot(x= 'YearsSinceLastPromotion' ,data = df ,palette='coolwarm_r'  )
plt.show()

# %%
sns.countplot(x= 'NumCompaniesWorked' ,data =df ,palette='winter_r',hue='Attrition')
plt.show()

# %% [markdown]
# > Employee who have completed 1 year have maximum attrition rate.

# %% [markdown]
# ## String encoding for dataset 

# %%
df.describe(include = "object")

# %%
cat_cols = ["BusinessTravel", "Department", "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobRole", "JobSatisfaction", "MaritalStatus", "OverTime", "PerformanceRating", "RelationshipSatisfaction", "WorkLifeBalance"]
# Dictionary to hold the LabelEncoder objects
label_encoders = {}

# Loop over the categorical columns and fit the LabelEncoder objects
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df[col])
    label_encoders[col] = le

# Dictionary to hold the mapping of label encoded values to original values
labels = {}

# Loop over the label encoders and add the mapping to the labels dictionary
for col, le in label_encoders.items():
    labels[col] = dict(zip(le.transform(le.classes_), le.classes_))

# Print the mapping of label encoded values to original values for each column
# print(labels)

label_mapping = {'BusinessTravel': {0: 'Non-Travel', 1: 'Travel_Frequently', 2: 'Travel_Rarely'},
                 'Department': {0: 'Human Resources', 1: 'Research & Development', 2: 'Sales'},
                 'Education': {0: ' Bachelor', 1: 'Below College', 2: 'College', 3: 'Doctor', 4: 'Master'},
                 'EducationField': {0: 'Human Resources', 1: 'Life Sciences', 2: 'Marketing', 3: 'Medical', 4: 'Other', 5: 'Technical Degree'},
                 'EnvironmentSatisfaction': {0: ' High', 1: 'Low ', 2: 'Medium', 3: 'Very High'},
                 'Gender': {0: 'Female', 1: 'Male'},
                 'JobInvolvement': {0: ' High', 1: 'Low ', 2: 'Medium', 3: 'Very High'},
                 'JobRole': {0: 'Healthcare Representative', 1: 'Human Resources', 2: 'Laboratory Technician', 3: 'Manager', 4: 'Manufacturing Director', 5: 'Research Director', 6: 'Research Scientist', 7: 'Sales Executive', 8: 'Sales Representative'},
                 'JobSatisfaction': {0: ' High', 1: 'Low ', 2: 'Medium', 3: 'Very High'},
                 'MaritalStatus': {0: 'Divorced', 1: 'Married', 2: 'Single'},
                 'OverTime': {0: 'No', 1: 'Yes'},
                 'PerformanceRating': {0: ' Excellent', 1: ' Outstanding'},
                 'RelationshipSatisfaction': {0: ' High', 1: 'Low ', 2: 'Medium', 3: 'Very High'},
                 'WorkLifeBalance': {0: ' Best', 1: ' Better', 2: 'Bad ', 3: 'Good'}}

# Swap keys and values of nested dictionaries
swapped_label_mapping = {k: {v2: v1 for v1, v2 in v.items()} for k, v in label_mapping.items()}

# Print the swapped label mapping
print(swapped_label_mapping)


# %%
Attrition_le=LabelEncoder()
BusinessTravel_le=LabelEncoder()
Department_le=LabelEncoder()
Education_le=LabelEncoder()
EducationField_le=LabelEncoder()
EnvironmentSatisfaction_le=LabelEncoder()
Gender_le=LabelEncoder()
JobInvolvement_le=LabelEncoder()
JobRole_le=LabelEncoder()
JobSatisfaction_le=LabelEncoder()
MaritalStatus_le=LabelEncoder()
Over18_le=LabelEncoder()
OverTime_le=LabelEncoder()
PerformanceRating_le=LabelEncoder()
RelationshipSatisfaction_le=LabelEncoder()
WorkLifeBalance_le=LabelEncoder()

# %%
df['Attrition'] = Attrition_le.fit_transform(df['Attrition'])
df['BusinessTravel'] = BusinessTravel_le.fit_transform(df['BusinessTravel'])
df['Department'] = Department_le.fit_transform(df['Department'])
df['Education'] = Education_le.fit_transform(df['Education'])
df['EducationField'] = EducationField_le.fit_transform(df['EducationField'])
df['EnvironmentSatisfaction'] = EnvironmentSatisfaction_le.fit_transform(df['EnvironmentSatisfaction'])
df['Gender'] = Gender_le.fit_transform(df['Gender'])
df['JobInvolvement'] = JobInvolvement_le.fit_transform(df['JobInvolvement'])
df['JobRole'] = JobRole_le.fit_transform(df['JobRole'])
df['JobSatisfaction'] = JobSatisfaction_le.fit_transform(df['JobSatisfaction'])
df['MaritalStatus'] = MaritalStatus_le.fit_transform(df['MaritalStatus'])
df['OverTime'] = OverTime_le.fit_transform(df['OverTime'])
df['PerformanceRating'] = PerformanceRating_le.fit_transform(df['PerformanceRating'])
df['RelationshipSatisfaction'] = RelationshipSatisfaction_le.fit_transform(df['RelationshipSatisfaction'])
df['WorkLifeBalance'] = WorkLifeBalance_le.fit_transform(df['WorkLifeBalance'])

# %%
# temp
df.describe().T

# %% [markdown]
# **This graphs gives us idea that all the remaining features are some how related to the Target Variable**

# %% [markdown]
# ## Splitting Training and Testing data

# %%
X=df.drop(['Attrition'],axis=1)
y=df['Attrition']

# %%
X_train ,X_test ,y_train ,y_test=train_test_split(X,y ,test_size=0.2,random_state=42)

# %%
X_train.shape, y_train.shape

# %%
X_test.shape, y_test.shape

# %%
X_train.columns

# %% [markdown]
# **Training Data and testing data seperated**

# %% [markdown]
# Train a Random Forest Classifier

# %%
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# %% [markdown]
# Make predictions on the test set

# %%
y_pred = rfc.predict(X_test)

# %% [markdown]
# Evaluate the model's performance

# %%
accuracy = accuracy_score(y_test, y_pred) *100
print(f'Accuracy: {accuracy:.2f}%\n')
print(classification_report(y_test, y_pred))

# %% [markdown]
# Calculate and plot the confusion matrix

# %%
cm = confusion_matrix(y_test, y_pred)
sns.set(font_scale=1) # Adjust font size
plt.figure(figsize=(6, 3))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix", fontsize=18)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()


# %% [markdown]
# Plot feature importance

# %%
feat_importance = pd.Series(rfc.feature_importances_, index=X.columns)
feat_importance_sorted = feat_importance.sort_values(ascending=False)

plt.figure(figsize=(11, 7))
sns.barplot(
    x=feat_importance_sorted.values,
    y=feat_importance_sorted.index,
    palette="YlOrBr_r"
)

plt.title("Feature Importance", fontsize=20)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Feature", fontsize=16)

# Customize the plot background color
plt.gca().set_facecolor('#cbcffc')  # Set the background color to a darker shade

plt.show()

# %% [markdown]
# Compute ROC curve and ROC area for each class

# %%
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# %%
plt.figure(figsize=(10,6)) # Set figure size
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver Operating Characteristic', fontsize=20)
plt.legend(loc="lower right", fontsize=14)
plt.show()


# %% [markdown]
# Plot Precision-Recall curve

# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall Curve')
plt.plot([0, 1], [0.5, 0.5],'r--', label='Random Guessing')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=18)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(loc="lower left", fontsize=12)
plt.show()

# %% [markdown]
# Compute average precision score

# %%
# Compute average precision score
avg_precision = average_precision_score(y_test, y_pred)

# Plot Precision-Recall curve
plt.figure(figsize=(10,6)) # Set figure size
step_kwargs = {'step': 'post'}
plt.step(recall, precision, color='blue', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(avg_precision), fontsize=20)
plt.show()

# %% [markdown]
# ## Creating Joblib file to web based prediction

# %%
import joblib

joblib.dump(rfc, './website/model.joblib')



