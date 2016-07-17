import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f1 = '../data/gender_age_train.csv'

df1 = pd.read_csv(f1)
print df1.head()
print 'Plotting histograms of age and age by gender...'
df1['age'].hist(normed=True)
plt.savefig('charts/age.png')
df1['age'].hist(by=df1['gender'], normed=True)
plt.savefig('charts/age_by_gender.png')
plt.clf()

print 'Descriptions of age and age grouped by gender...'
print df1['age'].describe()
print df1.groupby('gender').describe()
print 'Description of negative device_id numbers...'
print (df1['device_id'] < 0).describe()

df1['device_id_negative'] = np.where(df1['device_id'] < 0, 1, 0)
print 
print 'Negative device ID by age and gender...'
print df1.groupby('gender')['device_id_negative'].describe()
print pd.crosstab(df1['gender'], df1['device_id_negative'])
print pd.crosstab(df1['age'], df1['device_id_negative'])

df1['device_id_negative'].hist(normed=True)
plt.savefig('charts/device_id_negative.png')
df1['device_id_negative'].hist(by=df1['gender'], normed=True)
plt.savefig('charts/device_id_negative_by_gender.png')
plt.clf()

#  add device information
f2 = '../data/phone_brand_device_model.csv'
df2 = pd.read_csv(f2)
print df2.head()

print 'Look at age and gender by phone brand...'
df1j = df1.merge(df2, on='device_id')
print df1j['phone_brand'].describe()
df1j['age'].hist(by=df1j['phone_brand'], normed=True, figsize=(30, 30))
plt.savefig('charts/age_by_phone_brand.png')
plt.clf()
df1j['is_male'] = np.where(df1j['gender'] == 'M', 1, 0)
df1j.groupby('phone_brand')['is_male'].mean().plot(kind='bar', figsize=(30, 10), title='Percent Male by Phone Brand')
plt.savefig('charts/gender_by_phone_brand.png')