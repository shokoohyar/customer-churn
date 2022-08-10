import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_excel(r'C:\Users\home\Desktop\New folder\Churn_Modelling.xls')

shape = df.shape
print(shape)

head = df.head()
print(head)

df.info()

df.nunique()
print(df.nunique())

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

desc=df[['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']].describe()
print(desc)

sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
print(sizes)

fg = sns.catplot(data=df, kind='count', x='Geography', col='Exited', palette = "Pastel2")
fg.fig.subplots_adjust(top=0.9)
plt.show()

total = float(len(df))
ax = sns.countplot(x="Geography", hue="Exited", data=df, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by Geography', fontsize=10)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')
plt.show()

x,y = 'Geography', 'Exited'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by country', fontsize=10)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x, txt_y, txt)
plt.show()

total = float(len(df))
ax = sns.countplot(x="Gender", hue="Exited", data=df, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by Gender', fontsize=10)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')
plt.show()

x,y = 'Gender', 'Exited'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by each gender', fontsize=10)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x, txt_y, txt)
plt.show()

fg = sns.catplot(data=df, kind='count', x='Gender', col='Exited', palette = "Pastel2")
fg.fig.subplots_adjust(top=0.9)
plt.show()

total = float(len(df))
ax = sns.countplot(x="NumOfProducts", hue="Exited", data=df, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by NumOfProducts', fontsize=10)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')
plt.show()

x,y = 'NumOfProducts', 'Exited'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by each product', fontsize=10)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x, txt_y, txt)
plt.show()

fg = sns.catplot(data=df, kind='count', x='NumOfProducts', col='Exited', palette = "Pastel2")
fg.fig.subplots_adjust(top=0.9)
plt.show()

total = float(len(df))
ax = sns.countplot(x="HasCrCard", hue="Exited", data=df, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by HasCrCard', fontsize=10)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')
plt.show()

x,y = 'HasCrCard', 'Exited'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by HasCrCard', fontsize=10)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x, txt_y, txt)
plt.show()

fg = sns.catplot(data=df, kind='count', x='HasCrCard', col='Exited', palette = "Pastel2")
fg.fig.subplots_adjust(top=0.9)
plt.show()

total = float(len(df))
ax = sns.countplot(x="IsActiveMember", hue="Exited", data=df, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by IsActiveMember', fontsize=10)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')
plt.show()

x,y = 'IsActiveMember', 'Exited'
df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()
g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, palette = "Pastel2")
plt.title('percentage distribution of exited and retained customers by IsActiveMember', fontsize=10)
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()
    txt_y = p.get_height()
    g.ax.text(txt_x, txt_y, txt)
plt.show()

fg = sns.catplot(data=df, kind='count', x='IsActiveMember', col='Exited', palette = "Pastel2")
fg.fig.subplots_adjust(top=0.9)
plt.show()

CreditScore = np.array(df['CreditScore'])
fig,axis = plt.subplots(figsize=(8,6))
axis = sns.distplot(CreditScore,kde=False,bins=200)
plt.title('CreditScore distribution', fontsize=15)


g=sns.FacetGrid(df,hue = 'Exited',palette = "Pastel2")
(g.map(plt.hist,'CreditScore',edgecolor="w").add_legend())
plt.title('distribution of exited and retained customers by CreditScore ', fontsize=8)


plt.show()

age = np.array(df['Age'])
fig,axis = plt.subplots(figsize=(8,6))
axis = sns.distplot(age,kde=False,bins=200)
plt.title('age distribution', fontsize=15)


g=sns.FacetGrid(df,hue = 'Exited',palette = "Pastel2")
(g.map(plt.hist,'Age',edgecolor="w").add_legend())
plt.title('distribution of exited and retained customers by age ', fontsize=9)


plt.show()

Tenure = np.array(df['Tenure'])
fig,axis = plt.subplots(figsize=(8,6))
axis = sns.distplot(Tenure,kde=False,bins=200)
plt.title('Tenure distribution', fontsize=15)


g=sns.FacetGrid(df,hue = 'Exited',palette = "Pastel2")
(g.map(plt.hist,'Tenure',edgecolor="w").add_legend())
plt.title('distribution of exited and retained customers by Tenure ', fontsize=8)


plt.show()

Balance = np.array(df['Balance'])
fig,axis = plt.subplots(figsize=(8,6))
axis = sns.distplot(Balance,kde=False,bins=200)
plt.title('Balance distribution', fontsize=15)


g=sns.FacetGrid(df,hue = 'Exited',palette = "Pastel2")
(g.map(plt.hist,'Balance',edgecolor="w").add_legend())
plt.title('distribution of exited and retained customers by Balance ', fontsize=8)


plt.show()


EstimatedSalary= np.array(df['EstimatedSalary'])
fig,axis = plt.subplots(figsize=(8,6))
axis = sns.distplot(EstimatedSalary,kde=False,bins=200)
plt.title('EstimatedSalary distribution', fontsize=15)


g=sns.FacetGrid(df,hue = 'Exited',palette = "Pastel2")
(g.map(plt.hist,'EstimatedSalary',edgecolor="w").add_legend())
plt.title('distribution of exited and retained customers by EstimatedSalary ', fontsize=7)


plt.show()

# Decomposition predictors and target
predictors = df.iloc[:,0:10]
target = df.iloc[:,10:]

try:
    predictors['isMale'] = predictors['Gender'].map({'Male':1, 'Female':0})
except:
    pass

# Geography one shot encoder
predictors[['France', 'Germany', 'Spain']] = pd.get_dummies(predictors['Geography'])
print("here",predictors)
# Removal of unused columns.
predictors = predictors.drop(['Gender','Geography','Spain'], axis = 1)
print("there",predictors)




normalization = lambda x:(x-x.min()) / (x.max()-x.min())
transformColumns = predictors[["Balance","EstimatedSalary","CreditScore"]]
predictors[["Balance","EstimatedSalary","CreditScore"]] = normalization(transformColumns)

# All Predictors Columns
predictors.describe()

# Train and test splitting
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.25, random_state=0)
pd.DataFrame({"Train Row Count":[x_train.shape[0],y_train.shape[0]],
              "Test Row Count":[x_test.shape[0],y_test.shape[0]]},
             index=["X (Predictors)","Y (Target)"])

# Numpy excaptions handle
y_train = y_train.values.ravel()


# Logistic Regression
logr = LogisticRegression()
logr.fit(x_train,y_train)
y_pred_logr = logr.predict(x_test)
logr_acc = accuracy_score(y_test,y_pred_logr)

# Random Forrest
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
rfc_acc = accuracy_score(y_test,y_pred_rfc)

# Neural Network
nnc = MLPClassifier()
nnc.fit(x_train,y_train)
y_pred_nnc = nnc.predict(x_test)
nnc_acc = accuracy_score(y_test,y_pred_nnc)


print(pd.DataFrame({"Algorithms":["Logistic Regression","Random Ferest","Neural Network"],
              "Scores":[logr_acc,rfc_acc,nnc_acc]}))

print("done")



