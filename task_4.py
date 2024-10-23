import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt

# Hàm load hình ảnh từ thư mục
def load_images_from_folder(folder):
    images = []
    labels = []
    for label_folder in tqdm(os.listdir(folder), desc="Loading images from folder"):
        label_path = os.path.join(folder, label_folder)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(label_folder)
    return np.array(images), np.array(labels)

# Load tập dữ liệu từ thư mục 'flowers'
X, y = load_images_from_folder('flowers')

# Chuyển ảnh thành dạng vector 1D
X = X.reshape(X.shape[0], -1)

# Chuẩn hóa dữ liệu
X = X.astype('float32') / 255.0

# Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Giảm chiều dữ liệu với PCA
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Trạng thái khi huấn luyện các mô hình
print("Training models...")

# Huấn luyện và đánh giá mô hình SVM
print("Training SVM...")
start_time = time.time()
svm = SVC()
svm.fit(X_train_pca, y_train)
y_pred_svm = svm.predict(X_test_pca)
svm_time = time.time() - start_time
metrics_svm = evaluate_model(y_test, y_pred_svm)

# Huấn luyện và đánh giá mô hình KNN
print("Training KNN...")
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)
knn_time = time.time() - start_time
metrics_knn = evaluate_model(y_test, y_pred_knn)

# Huấn luyện và đánh giá mô hình Decision Tree
print("Training Decision Tree...")
start_time = time.time()
dt = DecisionTreeClassifier()
dt.fit(X_train_pca, y_train)
y_pred_dt = dt.predict(X_test_pca)
dt_time = time.time() - start_time
metrics_dt = evaluate_model(y_test, y_pred_dt)

# In kết quả đánh giá
print(f"SVM: Time = {svm_time:.4f}s, Accuracy = {metrics_svm[0]:.4f}, Precision = {metrics_svm[1]:.4f}, Recall = {metrics_svm[2]:.4f}, F1-Score = {metrics_svm[3]:.4f}")
print(f"KNN: Time = {knn_time:.4f}s, Accuracy = {metrics_knn[0]:.4f}, Precision = {metrics_knn[1]:.4f}, Recall = {metrics_knn[2]:.4f}, F1-Score = {metrics_knn[3]:.4f}")
print(f"Decision Tree: Time = {dt_time:.4f}s, Accuracy = {metrics_dt[0]:.4f}, Precision = {metrics_dt[1]:.4f}, Recall = {metrics_dt[2]:.4f}, F1-Score = {metrics_dt[3]:.4f}")

# Vẽ biểu đồ so sánh
models = ['SVM', 'KNN', 'Decision Tree']
accuracy = [metrics_svm[0], metrics_knn[0], metrics_dt[0]]
precision = [metrics_svm[1], metrics_knn[1], metrics_dt[1]]
recall = [metrics_svm[2], metrics_knn[2], metrics_dt[2]]
f1_score = [metrics_svm[3], metrics_knn[3], metrics_dt[3]]
times = [svm_time, knn_time, dt_time]

plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(models))

plt.bar(index, accuracy, bar_width, color='blue', label='Accuracy')
plt.bar(index + bar_width, precision, bar_width, color='green', label='Precision')
plt.bar(index + 2 * bar_width, recall, bar_width, color='red', label='Recall')
plt.bar(index + 3 * bar_width, f1_score, bar_width, color='purple', label='F1-Score')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.xticks(index + 1.5 * bar_width, models)
plt.legend()
plt.title('So sánh hiệu suất giữa SVM, KNN, và Decision Tree')
plt.show()

# Lưu các mô hình và PCA, LabelEncoder
print("Saving models...")
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt, f)

with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Models and other components saved successfully!")
