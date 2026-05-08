import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
ratings = pd.read_csv('ratings.csv')
movies  = pd.read_csv('movies.csv')

# Train CF model
reader   = Reader(rating_scale=(1, 5))
data     = Dataset.load_from_df(
    ratings[['user_id','item_id','rating']], reader)
trainset, _ = train_test_split(
    data, test_size=0.2, random_state=42)

model = SVD(n_factors=50, n_epochs=20, random_state=42)
model.fit(trainset)

# Extract and save ONLY the matrices
# (no surprise dependency needed to load!)
np.save('user_factors.npy',  model.pu)
np.save('item_factors.npy',  model.qi)
np.save('user_biases.npy',   model.bu)
np.save('item_biases.npy',   model.bi)

# Save global mean
global_mean = ratings['rating'].mean()
np.save('global_mean.npy', np.array([global_mean]))

# Save item id mapping
item_ids = sorted(ratings['item_id'].unique())
item_map  = {iid: idx for idx, iid in enumerate(item_ids)}
pd.DataFrame(list(item_map.items()),
             columns=['item_id','idx'])\
  .to_csv('item_map.csv', index=False)

print("✅ Model matrices saved!")
print(f"user_factors: {model.pu.shape}")
print(f"item_factors: {model.qi.shape}")
print(f"Global mean:  {global_mean:.3f}")