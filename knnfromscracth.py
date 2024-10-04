import numpy as np 
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')


x = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# 1. implement the euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

#2. find the k nearest neighbors
def get_neighbors(X_train, y_train, x_test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test_point)
        distances.append((dist, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

#3predict the class of the test point based on the majority vote of its neighbors
def predict_classification(neighbors):
    classes = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(classes).most_common(1)
    return most_common[0][0]

#4. full knn implementation

def knn(X_train, y_train, X_test, k):
    predictions = []
    for x_test_point in X_test:
        neighbors = get_neighbors(X_train, y_train, x_test_point, k)
        result = predict_classification(neighbors)
        predictions.append(result)
    return predictions

#5. manuell split the data into train and test sets 
def train_test_split_manual(x,y, test_size=0.2, rnd_state=None):
    data = np.column_stack([x,y])
    if rnd_state is not None:
        np.random.seed(rnd_state)
    np.random.shuffle(data)
    x_shuffled = data[:,:-1]
    y_shuffled = data[:,-1]
    
    split_index = int(len(x_shuffled) * (1-test_size))
    x_train = x_shuffled[:split_index]
    x_test = x_shuffled[split_index:]
    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split_manual(x,y, test_size=0.2, rnd_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#6. evaluate the accuracy 

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

k = 5
y_pred = knn(x_train, y_train, x_test, k)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

#7. normalize the data
def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

X_train_normalized = normalize(x_train)
X_test_normalized = normalize(x_test)

def evaluate_knn_for_different_k(x_train, y_train, X_test, y_test):
    accuracies = []
    k_values = range(1, 21)  # Test k-verdier fra 1 til 20
    for k in k_values:
        y_pred = knn(x_train, y_train, x_test, k)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f'k = {k}, Accuracy = {acc * 100:.2f}%')
    return accuracies

accuracies = evaluate_knn_for_different_k(X_train_normalized, y_train, X_test_normalized, y_test)

#8. implement precision, recall, f1-score, mean squared error and confusion matrix
def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Presisjon: {precision:.2f}')
print(f'Tilbakekalling: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

#9. implement confusion matrix
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TP, FP], [FN, TN]])


cm = confusion_matrix(y_test, y_pred)
print(f'Forvirringsmatrise:\n{cm}')

def plot_accuracies(k_values, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Nøyaktighet vs. K-verdi')
    plt.xlabel('K-verdi')
    plt.ylabel('Nøyaktighet')
    plt.grid(True)
    plt.show()

k_values = range(1, 21)
plot_accuracies(k_values, accuracies)
