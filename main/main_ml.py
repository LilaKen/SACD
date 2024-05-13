from utils.dataset import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time


class ClassifierWrapper:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, X, y):
        # Using tqdm just to show progress, not real epochs
        for _ in tqdm(range(10), desc="Training"):
            self.classifier.fit(X, y)
            time.sleep(0.1)  # Just to slow down tqdm for demonstration

    def evaluate(self, X, y):
        predictions = self.classifier.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy


fhg_fft = FHGFFT()
(train_data_fft, train_labels_fft), (test_data_fft, test_labels_fft) = fhg_fft.extract_data()
train_data_fft = train_data_fft.squeeze(axis=-1)
test_data_fft = test_data_fft.squeeze(axis=-1)

# 1. Decision Tree
clf_tree = ClassifierWrapper(DecisionTreeClassifier())
clf_tree.train(train_data_fft, train_labels_fft)
accuracy_tree = clf_tree.evaluate(test_data_fft, test_labels_fft)
print(f"Decision Tree Accuracy: {accuracy_tree * 100:.2f}%")

# 2. KNN
clf_knn = ClassifierWrapper(KNeighborsClassifier(n_neighbors=5))
clf_knn.train(train_data_fft, train_labels_fft)
accuracy_knn = clf_knn.evaluate(test_data_fft, test_labels_fft)
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")

# 3. Naive Bayes
clf_nb = ClassifierWrapper(GaussianNB())
clf_nb.train(train_data_fft, train_labels_fft)
accuracy_nb = clf_nb.evaluate(test_data_fft, test_labels_fft)
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")

# 4. SVM
clf_svm = ClassifierWrapper(SVC(kernel='linear', C=1, probability=True))
clf_svm.train(train_data_fft, train_labels_fft)
accuracy_svm = clf_svm.evaluate(test_data_fft, test_labels_fft)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
