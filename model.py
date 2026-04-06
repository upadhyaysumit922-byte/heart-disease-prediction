import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# Save
pickle.dump(model,  open('model.pkl',  'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("model.pkl and scaler.pkl saved!")