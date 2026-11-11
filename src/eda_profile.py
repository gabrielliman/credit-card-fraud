"""EDA profiling. Usage: conda run -n fraud python -m src.eda_profile data"""
import pandas as pd, numpy as np, sys
d=sys.argv[1]; tr=pd.read_csv(d+'/train.csv'); te=pd.read_csv(d+'/test.csv')
y='is_fraud'; print('nulls train/test', tr.isna().sum().sum(), te.isna().sum().sum())
print('dup rows (excl id)', tr.drop(columns=['id',y]).duplicated().sum())
print('target values', tr[y].value_counts().to_dict())
num=[c for c in tr.columns if c not in('id',y) and tr[c].dtype.kind in 'fi']
oth=[c for c in tr.columns if c not in num+['id',y]]
from sklearn.metrics import roc_auc_score
print('\n== numeric: describe + univariate AUC + train/test mean shift')
for c in num:
    a=roc_auc_score(tr[y],tr[c]); s=(te[c].mean()-tr[c].mean())/tr[c].std()
    print(f"{c:24s} min={tr[c].min():9.2f} med={tr[c].median():9.2f} max={tr[c].max():10.2f} nunq={tr[c].nunique():6d} AUC={max(a,1-a):.3f}{'+' if a>=.5 else '-'} shift={s:+.3f}")
print('\n== categorical/bool: fraud rate by level')
for c in oth:
    g=tr.groupby(c)[y].agg(['mean','size']).sort_values('mean',ascending=False)
    print(c, ' | '.join(f"{i}:{m:.3f}({n})" for i,(m,n) in g.iterrows()))
    print('   unseen test levels:', set(te[c].unique())-set(tr[c].unique()))
print('\n== adversarial validation (train vs test)')
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
X=pd.concat([tr.drop(columns=['id',y]),te.drop(columns=['id'])]); z=np.r_[np.zeros(len(tr)),np.ones(len(te))]
for c in oth: X[c]=X[c].astype('category')
print('AUC', cross_val_score(lgb.LGBMClassifier(n_estimators=200,verbose=-1),X,z,cv=3,scoring='roc_auc').mean())
print('\n== id vs target AUC', roc_auc_score(tr[y],tr['id']))
print('\n== top abs spearman corr between numerics')
cm=tr[num].corr('spearman').abs().where(lambda m: np.triu(np.ones(m.shape),1).astype(bool)).stack().sort_values(ascending=False); print(cm.head(8).to_string())
