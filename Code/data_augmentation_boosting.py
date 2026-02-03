import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import pearsonr
import warnings
import logging

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Ignore performance warnings
warnings.filterwarnings('ignore')

# Set up logging
print('Logging initialized for model training')

# Load the datasets
# mixture_definitions_df = pd.read_csv('./data/Extended_Mixture_Definitions.csv')
mixture_definitions_df = pd.read_csv('./data/Cleaned_Mixure_Definitions_Training_Set.csv')
# pred_percept_df = pd.read_csv('./data/pred_percept_single_gnn_164_ensemble.csv')
pred_percept_df = pd.read_csv('./data/weighted_embeddings_162.csv')

# Convert the 'prediction' column from string representation of list to actual list
pred_percept_df['prediction'] = pred_percept_df['prediction'].apply(eval)

# Create a dictionary with CID as key and its corresponding features as value
# features_dict = pred_percept_df.set_index('Unnamed: 0')['prediction'].to_dict()
features_dict = pred_percept_df.set_index('CID')['prediction'].to_dict()
# intensity_dict = pred_percept_df.set_index('Unnamed: 0')['INTENSITY'].to_dict()
intensity_dict = pred_percept_df.set_index('CID')['INTENSITY'].to_dict()


#################################################################################
# Weight computation using intensities
# exponential transformation
# all weights sum to 1, each weight is positive. 
# higher intensity -> larger weight
#################################################################################
def smooth_weights(numbers, tau=1.0):
    exp_numbers = np.exp(numbers / tau)
    weights = exp_numbers / np.sum(exp_numbers)
    return weights


def compute_average_features(row, tau):
    cids = row[2:]  # Skipping the first two columns (Dataset, Mixture Label)
    valid_features = []
    valid_intensities = []
    missing_cids = []

    for cid in cids:
        if cid != 0:
            if cid in features_dict:
                valid_features.append(features_dict[cid])
                valid_intensities.append(intensity_dict[cid])
            else:
                missing_cids.append(cid)

    if valid_features:
        smoothed_intensities = smooth_weights(np.array(valid_intensities), tau=tau) 
        weighted_features = np.average(valid_features, axis=0, weights=smoothed_intensities)
        average_features = weighted_features.tolist()
    else:
        average_features = [0] * 138  # Assuming 138 features

    return pd.Series([average_features, missing_cids], index=['average_features', 'missing_cids'])


def main(feature_construction, algorithm, data_augment, estim_num, augment_type, threshold):
    print(f"Feature Construction: {feature_construction}")
    print(f"Algorithm: {algorithm}")
    print(f"Data Augment: {data_augment}")
    print(f"Estimator Number: {estim_num}")
    print(f"Augment Type: {augment_type}")
    print(f"Threshold: {threshold}")

    # Grid search for the best combination of tau and n_estimators
    best_tau = None
    best_n_estimators = None
    best_pearson = -np.inf
    results = []

    # Load the appropriate training data
    if data_augment == 'yes':
        if augment_type == 'old':
            df_gt = pd.read_csv('./data/Extended_Training_Data_with_Larger_Noise.csv')
            mixture_definitions_df = pd.read_csv('./data/Extended_Mixture_Definitions.csv')
        elif augment_type == 'Iterative' or augment_type == 'Random':
            # df_gt = pd.read_csv(f'./data/{augment_type}_Training_Data_{threshold}.csv')
            df_gt = pd.read_csv(f'./data/New_{augment_type}_Training_Data_{threshold}.csv')
            # mixture_definitions_df = pd.read_csv(f'./data/{augment_type}_Definitions_{threshold}.csv')
            mixture_definitions_df = pd.read_csv(f'./data/New_{augment_type}_Definitions_{threshold}.csv')
    else:
        # df_gt = pd.read_csv('./data/training data.csv')
        df_gt = pd.read_csv('./data/Cleaned_Training_Data.csv')
        mixture_definitions_df = pd.read_csv('./data/Cleaned_Mixure_Definitions_Training_Set.csv')

    df_gt = df_gt.dropna()

    # Shuffle the data
    df_gt = df_gt.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Set the range for n_estimators based on estim_num
    if estim_num == 'high':
        n_estimators_range = range(200, 301, 10)
    elif estim_num == 'ultra':
        n_estimators_range = range(310, 501, 10)
    else:
        n_estimators_range = range(10, 191, 10)

    pear_all = []
    rmse_all = []
    models = []

    for tau in range(1, 51, 1):
        for n_estimators in n_estimators_range:
            # Apply the function to each row in the mixture_definitions_df
            result_df = mixture_definitions_df.apply(compute_average_features, axis=1, tau=tau)

            # Concatenate the results with the original mixture definitions
            final_df = pd.concat([mixture_definitions_df, result_df], axis=1)

            # Separate out the missing CIDs
            missing_cids = final_df.explode('missing_cids')[
                ['Dataset', 'Mixture Label', 'missing_cids']].dropna().reset_index(drop=True)

            # Define the feature names
            TASKS = [
                'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
                'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
                'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
                'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
                'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
                'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
                'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
                'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
                'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
                'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
                'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
                'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
                'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
                'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
                'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
                'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
                'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
                'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
                'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
                'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
            ]

            # Create a new dataframe with the correct column names
            new_dataset = final_df[['Dataset', 'Mixture Label']].copy()
            new_dataset[TASKS] = pd.DataFrame(final_df['average_features'].tolist(), index=final_df.index)

            # Prepare feature based on the selected method
            df_feature_list = []
            excluded_indices = []

            if data_augment == 'yes':
                for i in range(df_gt.shape[0]):
                    dataset1 = df_gt.iloc[i]['Dataset 1']
                    dataset2 = df_gt.iloc[i]['Dataset 2']
                    mixture1 = df_gt.iloc[i]['Mixture 1']
                    mixture2 = df_gt.iloc[i]['Mixture 2']
                    x = new_dataset[(new_dataset['Dataset'] == dataset1) & (new_dataset['Mixture Label'] == mixture1)]
                    y = new_dataset[(new_dataset['Dataset'] == dataset2) & (new_dataset['Mixture Label'] == mixture2)]
                    if not x.empty and not y.empty:
                        if feature_construction == 'square_difference':
                            z = (x.iloc[:, 2:].values.flatten() - y.iloc[:, 2:].values.flatten()) ** 2
                        elif feature_construction == 'absolute_difference':
                            z = np.abs(x.iloc[:, 2:].values.flatten() - y.iloc[:, 2:].values.flatten())
                        elif feature_construction == 'concat':
                            z = np.concatenate((x.iloc[:, 2:].values.flatten(), y.iloc[:, 2:].values.flatten()))
                        df_feature_list.append(z)
                    else:
                        excluded_indices.append(i)

            else:
                for i in range(df_gt.shape[0]):
                    dataset = df_gt.iloc[i]['Dataset']
                    mixture1 = df_gt.iloc[i]['Mixture 1']
                    mixture2 = df_gt.iloc[i]['Mixture 2']
                    x = new_dataset[(new_dataset['Dataset'] == dataset) & (new_dataset['Mixture Label'] == mixture1)]
                    y = new_dataset[(new_dataset['Dataset'] == dataset) & (new_dataset['Mixture Label'] == mixture2)]
                    if not x.empty and not y.empty:
                        if feature_construction == 'square_difference':
                            z = (x.iloc[:, 2:].values.flatten() - y.iloc[:, 2:].values.flatten()) ** 2
                        elif feature_construction == 'absolute_difference':
                            z = np.abs(x.iloc[:, 2:].values.flatten() - y.iloc[:, 2:].values.flatten())
                        elif feature_construction == 'concat':
                            z = np.concatenate((x.iloc[:, 2:].values.flatten(), y.iloc[:, 2:].values.flatten()))
                        df_feature_list.append(z)
                    else:
                        excluded_indices.append(i)

            df_feature = pd.DataFrame(df_feature_list)
            df_feature.reset_index(drop=True, inplace=True)
            df_gt = df_gt.drop(index=excluded_indices).reset_index(drop=True)

            X = df_feature.values
            y = df_gt['Experimental Values'].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)

            # Identify the original indices
            if data_augment == 'yes':
                if augment_type == 'old':
                    original_indices = np.arange(0, 2000, 4)  # 0, 4, 8, ..., 1996 (total 500 indices)
                    augmented_indices = np.setdiff1d(np.arange(2000), original_indices)  # Remaining indices
                elif augment_type == 'Iterative' or augment_type == 'Random':
                    labels = df_gt['label'].unique()
                    label_indices = np.arange(len(labels))
                    np.random.shuffle(label_indices)
                    kf = KFold(n_splits=10, shuffle=True, random_state=42)
            else:
                original_indices = np.arange(0, 500)
                labels = df_gt['label'].unique()
                label_indices = np.arange(len(labels))
                np.random.shuffle(label_indices)
                kf = KFold(n_splits=10, shuffle=True, random_state=42)

            for train_index, val_index in kf.split(labels):
                if augment_type == 'old':
                    val_indices = original_indices[val_index]
                    # Remove corresponding augmented indices from training set
                    augmented_to_remove = []
                    for val_idx in val_indices:
                        augmented_to_remove.extend([val_idx + i for i in range(1, 4)])
                    train_indices = np.setdiff1d(np.concatenate([original_indices[train_index], augmented_indices]),
                                                 augmented_to_remove)
                elif augment_type == 'Iterative' or augment_type == 'Random':
                    train_labels = labels[train_index]
                    val_labels = labels[val_index]

                    train_indices = df_gt[df_gt['label'].isin(train_labels)].index
                    val_indices = df_gt[df_gt['label'].isin(val_labels)].index
                else:
                    train_indices = original_indices[train_index]
                    val_indices = original_indices[val_index]

                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]

                if algorithm == 'xgboost':
                    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
                elif algorithm == 'lightgbm':
                    model = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=42)
                elif algorithm == 'catboost':
                    model = CatBoostRegressor(n_estimators=n_estimators, random_state=42, verbose=0)
                elif algorithm == 'randomforest':
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                model.fit(X_train, y_train)
                models.append(model)

                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                train_pearson = pearsonr(y_train, y_train_pred)[0]
                val_pearson = pearsonr(y_val, y_val_pred)[0]

                pear_all.append(val_pearson)
                rmse_all.append(val_rmse)

            avg_pearson = np.mean(pear_all)
            avg_rmse = np.mean(rmse_all)
            results.append((tau, n_estimators, avg_pearson, avg_rmse))

            print(
                f'Tau: {tau}, n_estimators: {n_estimators}, Average Pearson: {avg_pearson:.3f}, Average RMSE: {avg_rmse:.3f}')

            if avg_pearson > best_pearson:
                best_pearson = avg_pearson
                best_tau = tau
                best_n_estimators = n_estimators
                best_models = models
                print(f'New best model found with tau={tau}, n_estimators={n_estimators}, Pearson: {avg_pearson:.3f}')

    # Print the best result
    print(f'Best Tau: {best_tau}')
    print(f'Best n_estimators: {best_n_estimators}')
    print(f'Best Pearson Correlation: {best_pearson:.3f}')

    # Read the leaderboard submission form
    df_submission = pd.read_csv('./data/Leaderboard_set_Submission_form.csv')

    # Prepare features for the leaderboard submission form
    df_leaderboard_features = []

    for i in range(df_submission.shape[0]):
        dataset = df_submission.iloc[i]['Dataset']
        mixture1 = df_submission.iloc[i]['Mixture_1']
        mixture2 = df_submission.iloc[i]['Mixture_2']
        x = new_dataset[(new_dataset['Dataset'] == dataset) & (new_dataset['Mixture Label'] == mixture1)]
        y = new_dataset[(new_dataset['Dataset'] == dataset) & (new_dataset['Mixture Label'] == mixture2)]
        if not x.empty and not y.empty:
            if feature_construction == 'square_difference':
                z = (x.iloc[:, 2:].values.flatten() - y.iloc[:, 2:].values.flatten()) ** 2
            elif feature_construction == 'absolute_difference':
                z = np.abs(x.iloc[:, 2:].values.flatten() - y.iloc[:, 2:].values.flatten())
            elif feature_construction == 'concat':
                z = np.concatenate((x.iloc[:, 2:].values.flatten(), y.iloc[:, 2:].values.flatten()))
            df_leaderboard_features.append(z)
        else:
            df_leaderboard_features.append([np.nan] * X.shape[1])  # Handling missing values

    # Convert the list of features to a DataFrame
    df_leaderboard_features = pd.DataFrame(df_leaderboard_features)
    df_leaderboard_features.reset_index(drop=True, inplace=True)

    # Standardize leaderboard features
    df_leaderboard_features = scaler.transform(df_leaderboard_features)

    # Predict using the best models and average the predictions
    predictions = np.zeros(df_leaderboard_features.shape[0])

    for model in best_models:
        predictions += model.predict(df_leaderboard_features)

    predictions /= len(best_models)

    # Add predictions to the submission form
    df_submission['Predicted_Experimental_Values'] = predictions

    # Save the updated submission form
    output_submission_path = f'./val/{algorithm}_predictions_da_{data_augment}_{augment_type}_tau_{best_tau}_n_estimators_{best_n_estimators}_feature_construction_{feature_construction}_threshold_{threshold}.csv'
    df_submission.to_csv(output_submission_path, index=False)

    print("Predictions added and submission form saved.")

    # Read the Final submission form
    df_final = pd.read_csv('./data/Test_set_Submission_form.csv')

    # Prepare features for the leaderboard submission form
    df_final_features = []

    submission_mixture_definitions_df = pd.read_csv('./data/Mixure_Definitions_test_set.csv')

    # Apply the function to each row in the submission_mixture_definitions_df
    submission_result_df = submission_mixture_definitions_df.apply(compute_average_features, axis=1, tau=best_tau)

    # Concatenate the results with the original mixture definitions
    submission_final_df = pd.concat([submission_mixture_definitions_df, submission_result_df], axis=1)

    # Separate out the missing CIDs
    missing_cids = submission_final_df.explode('missing_cids')[
        ['Mixture Label', 'missing_cids']].dropna().reset_index(drop=True)

    # Define the feature names
    TASKS = [
        'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
        'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
        'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
        'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
        'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
        'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
        'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
        'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
        'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
        'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
        'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
        'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
        'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
        'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
        'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
        'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
        'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
        'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
        'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
        'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
    ]

    # Create a new dataframe with the correct column names
    submission_new_dataset = submission_final_df[['Mixture Label']].copy()
    submission_new_dataset[TASKS] = pd.DataFrame(submission_final_df['average_features'].tolist(),
                                                 index=submission_final_df.index)

    for i in range(df_final.shape[0]):
        mixture1 = df_final.iloc[i]['Mixture_1']
        mixture2 = df_final.iloc[i]['Mixture_2']
        x = submission_new_dataset[(submission_new_dataset['Mixture Label'] == mixture1)]
        y = submission_new_dataset[(submission_new_dataset['Mixture Label'] == mixture2)]
        if not x.empty and not y.empty:
            if feature_construction == 'square_difference':
                z = (x.iloc[:, 1:].values.flatten() - y.iloc[:, 1:].values.flatten()) ** 2
                # print(len(z))
            elif feature_construction == 'absolute_difference':
                z = np.abs(x.iloc[:, 1:].values.flatten() - y.iloc[:, 1:].values.flatten())
            elif feature_construction == 'concat':
                z = np.concatenate((x.iloc[:, 1:].values.flatten(), y.iloc[:, 1:].values.flatten()))
            df_final_features.append(z)
        else:
            df_final_features.append([np.nan] * X.shape[1])  # Handling missing values

    # Convert the list of features to a DataFrame
    df_final_features = pd.DataFrame(df_final_features)
    df_final_features.reset_index(drop=True, inplace=True)

    # Standardize leaderboard features
    df_final_features = scaler.transform(df_final_features)

    # Predict using the best models and average the predictions
    predictions = np.zeros(df_final_features.shape[0])

    for model in best_models:
        predictions += model.predict(df_final_features)

    predictions /= len(best_models)

    # Add predictions to the submission form
    df_final['Predicted_Experimental_Values'] = predictions

    # Save the updated submission form
    output_submission_path = f'./submit/final_submission_{algorithm}.csv'
    df_final.to_csv(output_submission_path, index=False)

    print("Final submission saved.")

args = argparse.Namespace(
    feature_construction='square_difference',
    algorithm='catboost',
    data_augment='no',
    estim_num='ultra',
    augment_type='Random',
    threshold=0.92
)

main(args.feature_construction, args.algorithm, args.data_augment, args.estim_num, args.augment_type, args.threshold)
