
categorical_cols = data2.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:\n", categorical_cols)

le = LabelEncoder()
data2['histological.type'] = le.fit_transform(data2['histological.type'])
hist_type_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Histological type mapping:", hist_type_mapping)

data2_encoded = pd.get_dummies(data2, columns=['PR.Status', 'ER.Status', 'HER2.Final.Status'], drop_first=True)

print("Shape after encoding:", data2_encoded.shape)

X = data2_encoded.drop('histological.type', axis=1)
y = data2_encoded['histological.type']

selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

retained_features = X.columns[selector.get_support()]

X_reduced_df = pd.DataFrame(X_reduced, columns=retained_features)

final_df = X_reduced_df.copy()
final_df['histological.type'] = y

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced_df.shape}")

sample_features = X_reduced_df.sample(n=50, axis=1, random_state=42)
corr_matrix = sample_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (Sampled 50 Features)")
plt.show()

corr_matrix_full = X_reduced_df.corr().abs()
upper = corr_matrix_full.where(np.triu(np.ones(corr_matrix_full.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

print(f"Number of features to drop due to high correlation: {len(to_drop)}")

X_uncorrelated = X_reduced_df.drop(columns=to_drop)

print(f"Shape after removing correlated features: {X_uncorrelated.shape}")

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_uncorrelated)

print(f"Shape after PCA (95% variance retained): {X_pca.shape}")

# Explained variance and cumulative variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot - PCA Components vs. Cumulative Variance')
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

# Check test set distribution
unique, counts = np.unique(y_test, return_counts=True)
print("Test set distribution:", dict(zip(unique, counts)))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_bal = scaler.fit_transform(X_train_bal)
X_test = scaler.transform(X_test)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Check new class distribution
unique, counts = np.unique(y_train_bal, return_counts=True)
print("Balanced class distribution:", dict(zip(unique, counts)))
