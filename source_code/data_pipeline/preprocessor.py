# Import thư viện StandardScaler từ scikit-learn để chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler
# Import thư viện pandas để thao tác trên DataFrame
import pandas as pd
# Import thư viện numpy để tính toán ma trận
import numpy as np
# Import thư viện typing để chú thích kiểu dữ liệu trả về
from typing import Tuple, List

# Hàm chuẩn hóa đặc trưng số học dựa trên phương pháp Z-score
def preprocess_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    # In thông báo bắt đầu bước chuẩn hóa đặc trưng
    print(f"[PREPROCESSOR] Bắt đầu chuẩn hóa {len(feature_cols)} đặc trưng được lựa chọn...")
    # Trích xuất các cột đặc trưng cần chuẩn hóa
    features_df = df[feature_cols].copy()
    # Kiểm tra và điền giá trị khuyết thiếu bằng giá trị trung vị của cột
    features_df = features_df.fillna(features_df.median())
    # Khởi tạo đối tượng StandardScaler để tính toán trung bình và độ lệch chuẩn
    scaler = StandardScaler()
    # Tính toán thông số và biến đổi tập dữ liệu thành phân phối chuẩn hóa
    scaled_array = scaler.fit_transform(features_df)
    # Tạo DataFrame mới với các cột và chỉ mục tương ứng
    scaled_df = pd.DataFrame(scaled_array, columns=feature_cols, index=df.index)
    # In thông báo hoàn thành chuẩn hóa
    print(f"[PREPROCESSOR] Chuẩn hóa thành công ma trận kích thước: {scaled_array.shape}")
    # Trả về mảng numpy, đối tượng scaler và DataFrame đã chuẩn hóa
    return scaled_array, scaler, scaled_df
