pip install feature-engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data Loading
crop = pd.read_csv(r'C:/Namit/Unified Mentor Internship/Crop production in India/Crop Production data.csv')
crop.info()
crop.describe()
crop.isnull().sum()            #production has 3730 null values
crop.dtypes


#EDA
mean_A = crop['Area'].mean()
mean_P = crop['Production'].mean()

median_A = crop['Area'].median()
median_P = crop['Production'].median()

mode_A = crop['Area'].mode()
mode_P = crop['Production'].mode()

range_A = max(crop['Area']) - min(crop['Area'])
range_P = max(crop['Production']) - min(crop['Production'])

var_A = crop['Area'].var()
var_P = crop['Production'].var()

std_P = crop['Area'].std()
std_P = crop['Production'].std()

skew_A = crop['Area'].skew()
skew_P = crop['Production'].skew()

kurt_A = crop['Area'].kurt()
kurt_P = crop['Production'].kurt()


#Data Preprocessing
#Missing values in Production column
max_P = max(crop['Production'])
min_P = min(crop['Production'])

from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
crop['Production'] = pd.DataFrame(median_imputer.fit_transform(crop[["Production"]]))

crop.isnull().sum() 


#Type Casting
crop['State_Name'] = crop['State_Name'].astype('str')
crop['District_Name'] = crop['District_Name'].astype('str')
crop['Season'] = crop['Season'].astype('str')
crop['Crop'] = crop['Crop'].astype('str')
crop['Crop_Year'] = crop['Crop_Year'].astype('int')


#Duplicates
dup = crop.duplicated()
sum(dup)                                   #no duplicates


#Outliers
#IQR method
sns.boxplot(crop.Area)
sns.boxplot(crop.Production)

IQR_A = crop['Area'].quantile(0.75) - crop['Area'].quantile(0.25)
lower_limitA = crop['Area'].quantile(0.25) - (IQR_A*1.5)
upper_limitA = crop['Area'].quantile(0.75) + (IQR_A*1.5)

outliers_A = np.where((crop['Area'] > upper_limitA) | (crop['Area'] < lower_limitA), True, False)
crop['Area_new'] =  np.where(crop['Area'] > upper_limitA, upper_limitA, np.where(crop['Area'] < lower_limitA, lower_limitA, crop['Area']))
sns.boxplot(crop.Area_new)

IQR_P = crop['Production'].quantile(0.75) - crop['Production'].quantile(0.25)
lower_limitP = crop['Production'].quantile(0.25) - (IQR_P*1.5)
upper_limitP = crop['Production'].quantile(0.75) + (IQR_P*1.5)

outliers_A = np.where((crop['Production'] > upper_limitP) | (crop['Production'] < lower_limitP), True, False)
crop['Production_new'] =  np.where(crop['Production'] > upper_limitP, upper_limitP, np.where(crop['Production'] < lower_limitP, lower_limitP, crop['Production']))
sns.boxplot(crop.Production_new)


#Label encoding and Standardization
from sklearn.preprocessing import StandardScaler, LabelEncoder
label_enc = LabelEncoder()
stan_sc = StandardScaler()


crop['State_Name_encoded'] = label_enc.fit_transform(crop['State_Name'])
crop['Season_encoded'] = label_enc.fit_transform(crop['Season'])
crop['Crop_encoded'] = label_enc.fit_transform(crop['Crop'])


crop[['Area', 'Production']] = stan_sc.fit_transform(crop[['Area', 'Production']])


#Data Distributon Visualization
crop['State_Name'].unique()
crop['Season'].unique()
crop['Crop'].unique()

df= pd.DataFrame(crop)

sns.barplot(x='Crop_Year', y='Production', hue='Crop', data=df)
plt.title('Crop Production by Year')
plt.xticks(rotation=45)
plt.show()

sns.lineplot(x="Crop_Year", y="Production", hue="Crop", data=df, marker="o")
plt.title('Production Trends by Crop Over the Years')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x="Season", y="Production", data=df)
plt.title('Distribution of Crop Production by Season')
plt.xticks(rotation=45)
plt.show()

sns.scatterplot(x="Area", y="Production", hue="Crop", style="Season", data=df, s=100)
plt.title('Area vs. Production for Different Crops')
plt.show()

crop_area = df.groupby('Crop')['Area'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=crop_area, x='Crop', y='Area', palette='viridis')
plt.title('Total Area per Crop')
plt.xlabel('Crop')
plt.ylabel('Total Area (in hectares)')
plt.xticks(rotation=45)
plt.show()