from feature_extractor import Stage1FeatureExtractor, FeatureConfig

extractor = Stage1FeatureExtractor(FeatureConfig())

# Single image
feat = extractor.extract("path/to/image.png")   # shape (57,)

# Batch — feed straight into sklearn
X = extractor.extract_batch(list_of_paths)       # shape (N, 57)
y = labels_4way                                  # {Real, LlamaGen, VAR/HMAR, RAR}

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000).fit(X, y)