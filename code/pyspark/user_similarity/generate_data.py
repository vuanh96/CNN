from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz, csc_matrix
import numpy as np

# row = np.array([0, 2, 2, 0, 1, 2])
# col = np.array([0, 0, 1, 2, 0, 3])
# data = list(np.random.rand(6))
# user_profiles = csc_matrix((data, (row, col)), shape=(5, 4))
# print(user_profiles.toarray())
# save_npz('user_profiles/user_profiles.npz', user_profiles)

data = load_npz('user_profiles/user_profiles.npz')
similar_matrix = cosine_similarity(data)  # similar by row
print(similar_matrix)