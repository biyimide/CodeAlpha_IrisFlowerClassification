import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#LOADING THE DATASETS
df= pd.read_csv('Iris.csv')

# EXplore the data
print(df.head())
print (df.describe())
print (df.shape)


#Data visualization#
# Pairplot
sns.pairplot(df.drop(columns=["Id"]),hue="Species", palette="Set2")
plt.suptitle("Pairplot - Iris Dataset", y= 1.02)
plt.savefig("iris_pairplot.png", bbox_inches="tight")
plt.close()

#correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.drop(columns=["Species", "Id"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Iris data Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_heatmap.png")
plt.close()

# box plot
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, feature in enumerate(features):
    ax = axes[idx // 2][idx % 2]
    sns.boxplot(x="Species", y=feature, hue="Species",
                data=df, palette="Set2", ax=ax, legend=False)
    ax.set_title(f"{feature} by Species")
    ax.set_xlabel("")
plt.suptitle("Feature Distribution by Species", fontsize=14)
plt.tight_layout()
plt.savefig("iris_boxplots.png")
plt.close()

print("Data visualization completed and saved as images.")

#Feature engineering
X = df.drop(columns=["Species"])  
y = df["Species"]  
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")   

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model training
model= KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
print("Model training completed.")

#Testing the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:") 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.show()
plt.savefig("iris_confusion_matrix.png")
plt.close() 
print("Confusion matrix saved as 'iris_confusion_matrix.png'.")

# making a prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)    
predicted_class = model.predict(sample_scaled)
predicted_species = le.inverse_transform(predicted_class)
print(f"Predicted species for sample {sample[0]}: {predicted_species[0]}")


# Save the model and scaler for future use
import joblib
joblib.dump(model, "iris_knn_model.joblib")
joblib.dump(scaler, "iris_scaler.joblib")
