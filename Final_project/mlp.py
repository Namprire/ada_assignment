!pip install datasets -q
from datasets import load_dataset

dataset = load_dataset("TimSchopf/arxiv_categories")
# Example mapping to 8 super categories (you can adapt or simplify further)
category_mapping = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "quant-ph": "Quantum Physics",
    "stat": "Statistics",
    "eess": "Electrical Engineering"
}

def map_label(categories):
    # Just pick the first label and extract its root
    primary = categories[0].split("->")[-1]
    for key in category_mapping:
        if primary.startswith(key):
            return category_mapping[key]
    return "Other"

# Apply mapping
for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(lambda x: {'label': map_label(x['categories'])})
    
!pip install -U sentence-transformers -q
# Encode titles with MiniLM on GPU
from sentence_transformers import SentenceTransformer
import torch

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model on GPU
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Encode titles — set device & batch size for speed
X_train = model.encode(dataset['train']['title'], 
                       batch_size=64, 
                       show_progress_bar=True, 
                       device=device)

X_test = model.encode(dataset['test']['title'], 
                      batch_size=64, 
                      show_progress_bar=True, 
                      device=device)

# Convert labels to numbers
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(dataset['train']['label'])
y_test = le.transform(dataset['test']['label'])
#Train MLPClassifier
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, early_stopping=True, random_state=42)
clf.fit(X_train, y_train)
#Evaluate
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()
#✅ Bonus: Class imbalance plot
import seaborn as sns
import pandas as pd

label_counts = pd.Series(dataset['train']['label']).value_counts()
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.xticks(rotation=45)
plt.title("Label Distribution (Train)")
plt.show()
