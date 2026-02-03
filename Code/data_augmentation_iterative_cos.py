import pandas as pd
import numpy as np
import random
from tqdm import tqdm  # Import tqdm for the progress bar

# Set random seed for replication
random.seed(42)

# Load the necessary files
# mixture_definitions_path = './data/Cleaned_Mixture_Definitions_Training_set.csv'
mixture_definitions_path = './data/Cleaned_Mixure_Definitions_Training_set.csv'
predictions_path = './data/ensemble_embeddings_161.csv'
training_data_path = './data/training data.csv'

mixture_definitions = pd.read_csv(mixture_definitions_path)
predictions = pd.read_csv(predictions_path)
training_data = pd.read_csv(training_data_path)

# Create a dictionary to map (dataset, mixture label) to their component single molecules
mixture_to_cids = {}

for index, row in mixture_definitions.iterrows():
    dataset_label = row['Dataset']
    mixture_label = row['Mixture Label']
    cids = row[2:].dropna().astype(int).tolist()  # Extract all CID columns and drop NaN values
    mixture_to_cids[(dataset_label, mixture_label)] = cids

# Function to find the nearest molecule based on cosine similarity, excluding itself
def find_most_similar_molecule_excluding_self(target_cid, predictions_df, threshold=0.99):
    # target_row = predictions_df[predictions_df['Unnamed: 0'] == target_cid]
    target_row = predictions_df[predictions_df['CID'] == target_cid]
    if target_row.empty:
        return None  # Return None if not found

    target_embedding = np.array(target_row['prediction'].values[0].strip('[]').split(', ')).astype(float)
    all_embeddings = predictions_df['prediction'].apply(
        lambda x: np.array(x.strip('[]').split(', ')).astype(float)).values

    similarities = np.array(
        [np.dot(target_embedding, emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(emb)) for emb in
         all_embeddings])

    # Exclude the target molecule itself
    # target_index = predictions_df[predictions_df['Unnamed: 0'] == target_cid].index[0]
    target_index = predictions_df[predictions_df['CID'] == target_cid].index[0]
    similarities[target_index] = -1

    most_similar_index = np.argmax(similarities)
    # most_similar_cid = predictions_df.iloc[most_similar_index]['Unnamed: 0']
    most_similar_cid = predictions_df.iloc[most_similar_index]['CID']
    max_similarity = similarities[most_similar_index]

    return most_similar_cid if max_similarity >= threshold else None

# Augment the training data
augmented_pairs = []
extended_mixture_definitions = mixture_definitions.copy()
label_column = []

# Start new mixture labels from 1 for the DA dataset
new_mixture_label = 1

# Add a progress bar to the data augmentation loop
label_count = 1
for index, row in tqdm(training_data.iterrows(), total=training_data.shape[0], desc="Augmenting data"):
    mixture_A = row['Mixture 1']
    mixture_B = row['Mixture 2']
    dataset = row['Dataset']
    experimental_value = row['Experimental Values']

    if (dataset, mixture_A) not in mixture_to_cids or (dataset, mixture_B) not in mixture_to_cids:
        continue

    cids_A = mixture_to_cids[(dataset, mixture_A)]
    cids_B = mixture_to_cids[(dataset, mixture_B)]

    # Filter out zero values (non-existent molecules)
    existing_cids_A = [cid for cid in cids_A if cid != 0]
    existing_cids_B = [cid for cid in cids_B if cid != 0]

    if not existing_cids_A or not existing_cids_B:
        continue

    # Iterate through each molecule in the mixture until a suitable replacement is found
    most_similar_cid_A = None
    most_similar_cid_B = None

    for cid_A in existing_cids_A:
        most_similar_cid_A = find_most_similar_molecule_excluding_self(cid_A, predictions)
        if most_similar_cid_A:
            break

    for cid_B in existing_cids_B:
        most_similar_cid_B = find_most_similar_molecule_excluding_self(cid_B, predictions)
        if most_similar_cid_B:
            break

    # If no suitable replacement found for either mixture, keep the original pair
    if most_similar_cid_A is None or most_similar_cid_B is None:
        augmented_pairs.append([dataset, dataset, mixture_A, mixture_B, experimental_value])
        label_column.append(label_count)
        label_count += 1
        continue

    # Create new mixtures by replacing only the selected molecule
    new_cids_A = [most_similar_cid_A if cid == cid_A else cid for cid in cids_A]
    new_cids_B = [most_similar_cid_B if cid == cid_B else cid for cid in cids_B]

    new_mixture_A_label = new_mixture_label
    new_mixture_B_label = new_mixture_label + 1
    new_mixture_label += 2

    mixture_to_cids[('DA', new_mixture_A_label)] = new_cids_A
    mixture_to_cids[('DA', new_mixture_B_label)] = new_cids_B

    # Add new pairs with "DA" as the dataset labels for the new mixtures
    augmented_pairs.append([dataset, dataset, mixture_A, mixture_B, experimental_value])
    augmented_pairs.append(['DA', dataset, new_mixture_A_label, mixture_B, experimental_value])
    label_column.append(label_count)
    label_column.append(label_count)
    # augmented_pairs.append([dataset, 'DA', mixture_A, new_mixture_B_label, experimental_value])
    # augmented_pairs.append(['DA', 'DA', new_mixture_A_label, new_mixture_B_label, experimental_value])
    # label_column.append(label_count)
    # label_column.append(label_count)

    # Add new mixtures to extended mixture definitions with dataset "DA"
    new_row_A = pd.Series(
        ['DA', new_mixture_A_label] + new_cids_A + [0] * (extended_mixture_definitions.shape[1] - 2 - len(new_cids_A)),
        index=extended_mixture_definitions.columns)
    new_row_B = pd.Series(
        ['DA', new_mixture_B_label] + new_cids_B + [0] * (extended_mixture_definitions.shape[1] - 2 - len(new_cids_B)),
        index=extended_mixture_definitions.columns)

    extended_mixture_definitions = pd.concat([extended_mixture_definitions, new_row_A.to_frame().T], ignore_index=True)
    extended_mixture_definitions = pd.concat([extended_mixture_definitions, new_row_B.to_frame().T], ignore_index=True)
    label_count += 1

# Create the extended training data dataframe
extended_training_data = pd.DataFrame(augmented_pairs, columns=['Dataset 1', 'Dataset 2', 'Mixture 1', 'Mixture 2',
                                                                'Experimental Values'])
extended_training_data['label'] = label_column

# Save the extended mixture definitions and extended training data to new CSV files
extended_mixture_definitions_path = './data/New_Iterative_Definitions_0.99.csv'
extended_training_data_path = './data/New_Iterative_Training_Data_0.99.csv'

extended_mixture_definitions.to_csv(extended_mixture_definitions_path, index=False)
extended_training_data.to_csv(extended_training_data_path, index=False)

extended_mixture_definitions_path, extended_training_data_path
