import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
import pickle

path0 = './data/'
path1 = './model_single/'

# 575 by 4869
df_dragon = pd.read_csv(path0 + 'Dragon_Descriptors.csv', index_col=0)

# 338 by 21
df_percept = pd.read_csv(path0 + 'percept_single.csv', index_col=0)

df4 = pd.read_csv(path0 + 'Cleaned_Mixure_Definitions_Training_Set.csv', index_col=0)

cid_columns = [col for col in df4.columns if col.startswith('CID')]

id_all = pd.unique(df4[cid_columns].values.ravel())

id_all = np.append(id_all, 650)

print(type(id_all))

print(id_all)

print(len(id_all))

id_all = id_all[~np.isnan(id_all)]
id_all = id_all[id_all > 0]
id_all = id_all.astype('int')
# 165

print(len(id_all))

id_all_set = set(id_all)
df_dragon_index_set = set(df_dragon.index.astype(int))

id_exclude = [the_id for the_id in id_all_set if the_id not in df_dragon_index_set]

print(id_exclude)

id_all_set -= set(id_exclude)

id_all = list(id_all_set)

print(len(id_all))

# id_exclude = []
# for the_id in id_all:
#     if the_id not in df_dragon.index:
#         print(the_id)
#         id_exclude += [the_id]
# #7284
# #11173
# #84682
# #5284503
# #5318042
# #11002307
#
# for the_id in id_exclude:
#     id_all.remove(the_id)
#
# len(id_all)
# #159

######################################################################
# loop all the molecules in dragon dataset (for each molecule)
# for each perceptual attribute, use the trained ElasticNet model to assign a scalar value for the molecule for this perceptual attribute
# stack the predictions (scalars) column-wise to form the embedding vector
######################################################################

X = df_dragon.loc[id_all, :].to_numpy()
X = np.nan_to_num(X)
pred_all = []
for i in range(df_percept.shape[1]):
    # print(i)
    name_model = path1 + str(i) + '_' + df_percept.columns[i]
    the_model = pickle.load(open(name_model, 'rb'))
    pred = the_model.predict(X)
    pred_all += [pred]

pred_all = np.array(pred_all)
pred_all = pred_all.T

df_pred = pd.DataFrame(data=pred_all)
df_pred.columns = df_percept.columns
df_pred.index = id_all
#df_pred['CID'] = id_all
#df_pred = df_pred[['CID'] + df_percept.columns.tolist()]

df_pred.to_csv(path0 + 'Cleaned_pred_percept_single_162.csv')

######################################################################
# 
###################################################################### 


#############################
## extract features for mixture

# the_df = df4
# mixture_all = []
# for i in range(the_df.shape[0]):
#     feature_all = []
#     for j in range(the_df.shape[1]):
#         if the_df.iloc[i, j] > 0:
#             the_id = int(the_df.iloc[i, j])
#             if the_id in df_pred.index:
#                 feature_all += [df_pred.loc[the_id, :].tolist()]
#     feature_all = np.array(feature_all)
#     mixture_all += [(np.mean(feature_all, axis=0)).tolist()]
#
# mixture_all = np.array(mixture_all)
# print(mixture_all.shape)
# df_mixture = pd.DataFrame(data=mixture_all)
# df_mixture.columns = df_percept.columns
#
# df_mixture['Mixture Label'] = df4['Mixture Label'].values
# df_mixture.index = the_df.index
#
# columns = ['Mixture Label'] + df_percept.columns.tolist()
# df_mixture = df_mixture[columns]
#
# df_mixture.to_csv(path0 + 'percept_mixture_challenge.csv')
#
# print(df_mixture.head())