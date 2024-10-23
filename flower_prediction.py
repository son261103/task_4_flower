import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pickle

# Hàm load mô hình đã huấn luyện từ file
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Hàm load hình ảnh từ đường dẫn để dự đoán
def load_image_for_prediction(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))  # Resize hình ảnh giống như khi huấn luyện
        img = img.astype('float32') / 255.0  # Chuẩn hóa giống như dữ liệu huấn luyện
        img = img.flatten()  # Chuyển thành vector 1D
        return img
    else:
        print("Không thể load ảnh từ đường dẫn:", img_path)
        return None

# Tải mô hình và PCA đã huấn luyện
pca = load_model('pca_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
dt_model = load_model('dt_model.pkl')

# Tải Label Encoder để chuyển đổi lại nhãn từ số sang tên loài hoa
label_encoder = load_model('label_encoder.pkl')

# Dự đoán loại hoa
def predict_flower(img_path, model, pca, label_encoder):
    img_vector = load_image_for_prediction(img_path)
    if img_vector is not None:
        # Giảm chiều dữ liệu với PCA
        img_pca = pca.transform([img_vector])
        # Dự đoán nhãn
        predicted_label = model.predict(img_pca)
        # Chuyển nhãn số thành tên loài hoa
        flower_name = label_encoder.inverse_transform(predicted_label)[0]
        return flower_name
    return None

# Đường dẫn ảnh cần dự đoán
img_path = '118974357_0faa23cce9_n.jpg'  # Thay đổi đường dẫn thành ảnh bạn muốn dự đoán

# Chọn mô hình bạn muốn sử dụng (SVM, KNN, hoặc Decision Tree)
model_to_use = svm_model  # Bạn có thể đổi thành knn_model hoặc dt_model

# Thực hiện dự đoán
flower_name = predict_flower(img_path, model_to_use, pca, label_encoder)
if flower_name:
    print(f"Loài hoa dự đoán: {flower_name}")
else:
    print("Không thể dự đoán loại hoa.")
