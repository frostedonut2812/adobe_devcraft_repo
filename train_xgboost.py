import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
pd.set_option('display.max_columns', None)

print("Libraries imported")

with open('Adobe Devcraft PS/user.profile.tags.txt') as f:
    tag_dict = {}
    for idx, line in enumerate(f.readlines()):
        tag_dict[line[:5]] = line[6:-1]

column_names = [
    "BidID", "Timestamp", "Logtype", "VisitorID", "User-Agent", "IP", "Region", "City",
    "Adexchange", "Domain", "URL", "AnonymousURLID", "AdslotID", "Adslotwidth",
    "Adslotheight", "Adslotvisibility", "Adslotformat", "Adslotfloorprice",
    "CreativeID", "Biddingprice", "Payingprice", "KeypageURL", "AdvertiserID", "User_tag"
]

clk = pd.read_csv("dataset/clk.08.txt", delimiter='\t',names=column_names ,low_memory=False)
conv = pd.read_csv("dataset/conv.08.txt", delimiter='\t',names=column_names, low_memory=False)
imp = pd.read_csv("dataset/imp.08.txt", delimiter='\t',names=column_names, low_memory=False)
print("Files loaded")
print(f"Imporession txt shape : {imp.shape}")
print(f"Click txt shape       : {clk.shape}")
print(f"Conversion txt shape  : {conv.shape}")

imp['got_clk'] = imp['BidID'].isin(clk['BidID'])
imp['got_conv'] = imp['BidID'].isin(conv['BidID'])
del clk, conv

imp = imp.drop(['Logtype', 'VisitorID', 'User-Agent', 'AdslotID', 'IP', 'Adexchange', 'Domain', 'URL', 'AnonymousURLID', 'Payingprice', 'Adslotfloorprice', 'Biddingprice', 'KeypageURL', 'Timestamp'], axis=1)

print("OHE encoding user tags....")
imp['User_tag'] = imp['User_tag'].str.split(',')
imp = imp.explode('User_tag')
imp = pd.get_dummies(imp, columns=['User_tag'])
imp = imp.groupby('BidID', as_index=False).max()
column_names = list(imp.columns)
for i, col in enumerate(column_names):
    if(not col[:4] == 'User'):
        continue
    column_names[i] = tag_dict[col[-5:]]
imp.columns = column_names
for key in tag_dict:
    if(tag_dict[key] not in column_names):
        imp[tag_dict[key]] = False

# label encoding
print("Label encoding categorical features...")
label_encoders = {}
imp = imp.drop(columns=['BidID'], axis=1)
categorical_features = ['Region', 'City', 'Adslotvisibility', 'Adslotformat', 'CreativeID', 'AdvertiserID']
for col in categorical_features:
    le = LabelEncoder()
    imp[col] = le.fit_transform(imp[col])
    label_encoders[col] = le

boolean_features = [col for col in imp.columns if imp[col].dtype == 'bool']
imp[boolean_features] = imp[boolean_features].astype(int)

# oversampling
print("Oversampling...")
clk_class = imp[imp['got_clk'] == 1]
got_clk_resample = resample(clk_class, replace=True, n_samples=700000)
imp = pd.concat([imp, got_clk_resample])

y = imp['got_conv']
X = imp.drop(columns=['got_conv'])
smote = SMOTE(sampling_strategy=0.15)
X, y = smote.fit_resample(X, y)
imp = pd.DataFrame(X, columns=X.columns)
imp['got_conv'] = y

print("Shape:", imp.shape)
print("Number of clicked impressions:",imp['got_clk'].sum())
print("Number of converted impressions:",imp['got_conv'].sum())

# separating labels
X = imp.drop(['got_conv', 'got_clk'], axis=1)
y_clk = imp['got_clk']
y_conv = imp['got_conv']


# TRAINING MODELS
print("Training our models.....")
# training clk model
print("Clk model....")
X_train, X_test, y_train, y_test = train_test_split(X, y_clk, test_size=0.1, stratify=y_clk)
clk_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False, n_estimators=60, max_depth=6)
clk_model.fit(X_train, y_train)

# Evaluate
clk_pred_prob = clk_model.predict_proba(X_test)[:, 1]  # Get probability of got_clk = 1
auc_score = roc_auc_score(y_test, clk_pred_prob)
print(f"Click Model AUC: {auc_score}")
clk_pred = clk_model.predict(X_test)
cm1 = confusion_matrix(y_test, clk_pred)
print("Clk Confusion matrix:",cm1)

# training conv model
print("Conv model...")
X_train, X_test, y_train, y_test = train_test_split(X, y_conv, test_size=0.1, stratify=y_conv)
conv_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False, n_estimators=60, max_depth=6)
conv_model.fit(X_train, y_train)

# Evaluate
conv_pred_prob = conv_model.predict_proba(X_test)[:, 1]  # Get probability of got_conv = 1
auc_score = roc_auc_score(y_test, conv_pred_prob)
print(f"Conv Model AUC: {auc_score}")
conv_pred = conv_model.predict(X_test)
cm2 = confusion_matrix(y_test, conv_pred)
print("Conv confusion matrix:",cm2)

# saving state dicts
proba_state_dicts = {
    "clk_model": clk_model,
    "conv_model": conv_model,
    "label_encoder": label_encoders
}
joblib.dump(proba_state_dicts, "save_dicts/xgboost_proba.joblib")
print("Dictionaries saved")
print("DONE")