import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nn_utilities import *
import os

######################### Data loading and pre-processing######################

col_names = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'outcome2'
]

path = "./NSL_KDD/"
training_filename_read = os.path.join(path,"KDDTrain+.csv")
test_filename_read = os.path.join(path,"KDDTest+.csv")

df=pd.read_csv(training_filename_read, header=None, names = col_names)
df_test = pd.read_csv(test_filename_read, header=None, names = col_names)
#drop rows that have missing data
df.dropna(inplace=True,axis=1)

print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)

#drop unused columns
#df['num_outbound_cmds'].value_counts()
#df_test['num_outbound_cmds'].value_counts()

df.drop('num_outbound_cmds', axis=1, inplace=True)
df_test.drop('num_outbound_cmds', axis=1, inplace=True)
df.drop('outcome2', axis=1, inplace=True)
df_test.drop('outcome2', axis=1, inplace=True)


print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())

print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

categorical_columns=['protocol_type', 'service', 'flag'] 

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values

unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]

unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]

unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]

dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)


unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc)

testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)

testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data

trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference
for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape

#drop the columns that are replaced by the dummy variable
newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)

newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)

labeldf=newdf['outcome']
labeldf_test=newdf_test['outcome']

newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1
                           ,'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1
                           ,'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1})

newdf['outcome'] = newlabeldf
newdf_test['outcome'] = newlabeldf_test
print(newdf['outcome'])
#print the final number of features in both training, and testing dataset
len(newdf.columns)
len(newdf_test.columns)

Y = newdf.outcome
Y_test = newdf_test.outcome

print(Y.shape)
print(Y_test.shape)

#drop the outcome column from the dataset
X = newdf.drop('outcome',1)
X_test = newdf_test.drop('outcome',1)

print(X.shape)
print(X_test.shape)

colNames=list(X)
colNames_test=list(X_test)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

sX = minmax_scale(X, axis = 0)
ncol = sX.shape[1]

####################################### Build Auto Encoder Network on training Data##############
#start auto-encoding of the training data
X_train, X_tes, Y_train, Y_tes = train_test_split(sX, Y, train_size = 1, random_state = seed(2019))
input_dim = Input(shape = (ncol, ))
encoding_dim = 30
encoded1 = Dense(100, activation = 'relu')(input_dim)
encoded2 = Dense(80, activation = 'relu')(encoded1)
encoded3 = Dense(50, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

decoded1 = Dense(30, activation = 'relu')(encoded4)
decoded2 = Dense(50, activation = 'relu')(decoded1)
decoded3 = Dense(80, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)

#fit encoder to the training data
autoencoder = Model(input = input_dim, output = decoded4)
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch = 100, batch_size = 100, shuffle = True, validation_data = (X_tes, X_tes))
#decoding the test data
encoder = Model(input = input_dim, output = encoded4)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_tes)


len(encoded_out.shape)
from sklearn.svm import SVC
svcl=SVC(kernel="linear")
svcl.fit(encoded_out,Y_tes)


#scaling X of test set
sX1 = minmax_scale(X_test, axis = 0)
ncol1 = sX1.shape[1]

############################ Encode the unseen data with auto encoder then predict using classifier

X_trai, X_te, Y_trai, Y_te = train_test_split(sX1, Y_test, train_size = 1, random_state = seed(2017))


input_dimn = Input(shape = (ncol, ))

encoding_dimn = 30
encoded1 = Dense(100, activation = 'relu')(input_dimn)
encoded2 = Dense(80, activation = 'relu')(encoded1)
encoded3 = Dense(50, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

decoded1 = Dense(30, activation = 'relu')(encoded4)
decoded2 = Dense(50, activation = 'relu')(decoded1)
decoded3 = Dense(80, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)

autoencoder = Model(input = input_dimn, output = decoded4)

autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_trai, X_trai, nb_epoch = 100, batch_size = 100, shuffle = True, validation_data = (X_te, X_te))

encoder = Model(input = input_dimn, output = encoded4)
encoded_input = Input(shape = (encoding_dimn, ))
encoded_outn = encoder.predict(X_te)

y_pred=svcl.predict(encoded_outn)
from sklearn.metrics import accuracy_score
print (accuracy_score(Y_te,y_pred))

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(Y_te,y_pred))
print(classification_report(Y_te,y_pred))
print(accuracy_score(Y_te,y_pred))
#####################
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_tes,Y_tes)

clf.predict(X_te)
Y_pred=clf.predict(X_te)

pd.crosstab(Y_te, Y_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])

y_pred=clf.predict(X_te)
from sklearn.metrics import accuracy_score
print (accuracy_score(Y_te,y_pred))

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(Y_te,y_pred))
print(classification_report(Y_te,y_pred))
print(accuracy_score(Y_te,y_pred))

from sklearn.metrics import classification_report
report = classification_report(Y_te, y_pred)
print (report)
