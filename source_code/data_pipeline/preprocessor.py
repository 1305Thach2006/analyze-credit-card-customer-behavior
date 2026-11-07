# Import thu vien StandardScaler tu scikit-learn de chuan hoa du lieu
from sklearn.preprocessing import StandardScaler
# Import thu vien pandas de thao tac tren DataFrame
import pandas as pd
# Import thu vien numpy de tinh toan ma tran
import numpy as np
# Import thu vien typing de chu thich kieu du lieu tra ve
from typing import Tuple, List

# Ham chuan hoa dac trung so hoc dua tren phuong phap Z-score
def preprocess_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    # In thong bao bat dau buoc chuan hoa dac trung
    print(f"[PREPROCESSOR] Bat dau chuan hoa {len(feature_cols)} dac trung duoc lua chon...")
    # Trich xuat cac cot dac trung can chuan hoa
    features_df = df[feature_cols].copy()
    # Kiem tra va dien gia tri khuyet thieu bang gia tri trung vi cua cot
    features_df = features_df.fillna(features_df.median())
    # Khoi tao doi tuong StandardScaler de tinh toan trung binh va do lech chuan
    scaler = StandardScaler()
    # Tinh toan thong so va bien doi tap du lieu thanh phan phoi chuan hoa
    scaled_array = scaler.fit_transform(features_df)
    # Tao DataFrame moi voi cac cot va chi muc tuong ung
    scaled_df = pd.DataFrame(scaled_array, columns=feature_cols, index=df.index)
    # In thong bao hoan thanh chuan hoa
    print(f"[PREPROCESSOR] Chuan hoa thanh cong ma tran kich thuoc: {scaled_array.shape}")
    # Tra ve mang numpy, doi tuong scaler va DataFrame da chuan hoa
    return scaled_array, scaler, scaled_df
