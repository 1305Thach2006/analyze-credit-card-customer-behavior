# Import thu vien os de thao tac voi thu muc va duong dan tep
import os
# Import thu vien sys de cau hinh duong dan module trong he thong
import sys
# Import thu vien pandas de thao tac va quan ly bang du lieu
import pandas as pd
# Import thu vien numpy de tinh toan ma tran va mang so hoc
import numpy as np
# Import thu vien StandardScaler tu scikit-learn de chuan hoa du lieu
from sklearn.preprocessing import StandardScaler

# Them thu muc hien tai vao duong dan tim kiem module cua Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import cac thiet lap cau hinh tu config/settings.py
from config.settings import (
    DATA_PATH, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
    RANDOM_STATE, N_CLUSTERS, SPENDING_CATEGORIES, RATIO_CATEGORIES,
    BASE_REWARD_MULTIPLIERS, EXPENSE_BONUS_FACTORS, RISK_THRESHOLDS
)

# Import ham nap du lieu tu data_pipeline/loader.py
from data_pipeline.loader import load_credit_card_data
# Import ham tao dac trung nghiep vu tu data_pipeline/feature_engineering.py
from data_pipeline.feature_engineering import engineer_credit_features
# Import ham chuan hoa du lieu tu data_pipeline/preprocessor.py
from data_pipeline.preprocessor import preprocess_features

# Import cac ham ve bieu do tu eda/visualizer.py
from eda.visualizer import (
    plot_correlation_heatmap, plot_category_spending_boxplot,
    plot_pca_clusters, plot_model_benchmark
)

# Import bo phan cum khach hang K-Means va PCA tu models/clustering.py
from models.clustering import CustomerClustering
# Import bo tinh diem va phan tich hanh vi tu models/recommender.py
from models.recommender import CustomerBehaviorScorer
# Import dong co kiem chung mo hinh Machine Learning tu models/validation_models.py
from models.validation_models import RewardValidationEngine

# Ham luu DataFrame ra tep CSV an toan tranh loi PermissionError khi tep dang mo trong Excel
def save_dataframe_safely(df: pd.DataFrame, file_path: str) -> str:
    # Thu thuc hien ghi tep CSV binh thuong
    try:
        # Ghi du lieu ra tap tin CSV voi ma hoa utf-8-sig
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        # In thong bao luu tep thanh cong
        print(f'[REPORT] Da luu thanh cong bao cao tai: {file_path}')
        # Tra ve duong dan tep da luu
        return file_path
    # Bat loi khi tep tin dang bi khoa boi ung dung khac nhu Microsoft Excel
    except PermissionError:
        # Tach duong dan thanh phan goc va duoi mo rong
        base_name, ext = os.path.splitext(file_path)
        # Tao ten tep thay the voi hau to _updated de khong lam gian doan chuong trinh
        alt_path = f'{base_name}_updated{ext}'
        # Ghi du lieu ra ten tep thay the
        df.to_csv(alt_path, index=False, encoding='utf-8-sig')
        # In thong bao tep goc dang mo trong Excel va da luu ra tep moi
        print(f'[CANH BAO] Tap tin {os.path.basename(file_path)} dang mo trong Excel. Da luu vao: {alt_path}')
        # Tra ve duong dan tep moi
        return alt_path

# Ham dieu phoi toan bo luong phan tich du lieu va mo hinh hoa
def main():
    # In duong ke ngan cach tieu de
    print('=' * 80)
    # In tieu de quy trinh phan tich
    print('  MO HINH PHAN TICH HANH VI TIEU DUNG THE TIN DUNG VA DE XUAT CHIEN LUOC UU DAI CA NHAN')
    # In duong ke ngan cach
    print('=' * 80)

    # -------------------------------------------------------------
    # BUOC 1: NAP DU LIEU (DATA INGESTION)
    # -------------------------------------------------------------
    # Goi ham doc du lieu tu tep CSV goc
    raw_df = load_credit_card_data(DATA_PATH)

    # -------------------------------------------------------------
    # BUOC 2: KY THUAT TAO DAC TRUNG (FEATURE ENGINEERING)
    # -------------------------------------------------------------
    # Goi ham tinh toan ty trong chi tieu va co canh bao rui ro
    df_engineered = engineer_credit_features(raw_df)

    # -------------------------------------------------------------
    # BUOC 3: PHAN TICH THONG KE MO TA VA TRUC QUAN HOA (EDA)
    # -------------------------------------------------------------
    # In thong bao buoc truc quan hoa du lieu
    print('\n[BUOC 3] Thuc hien truc quan hoa du lieu EDA...')
    # Danh sach cac cot dai dien can ve ma tran tuong quan
    corr_cols = [
        'Monthly_Spending', 'Annual_Income', 'Credit_Limit', 'Credit_Utilization',
        'Credit_Score', 'Payment_Ratio', 'Redemption_Rate', 'Discretionary_Ratio'
    ]
    # Duong dan tap tin anh heatmap tuong quan
    heatmap_file = os.path.join(FIGURES_DIR, 'eda_correlation_heatmap.png')
    # Ve va luu bieu do ma tran he so tuong quan
    plot_correlation_heatmap(df_engineered, corr_cols, heatmap_file)

    # Duong dan tap tin bieu do hop phan phoi chi tieu
    boxplot_file = os.path.join(FIGURES_DIR, 'category_spending_boxplot.png')
    # Ve va luu bieu do hop cho 7 danh muc chi tieu
    plot_category_spending_boxplot(df_engineered, SPENDING_CATEGORIES, boxplot_file)

    # -------------------------------------------------------------
    # BUOC 4: TIEN XU LY VA CHUAN HOA DU LIEU (PREPROCESSING)
    # -------------------------------------------------------------
    # In thong bao buoc tien xu ly chuan hoa
    print('\n[BUOC 4] Chuan hoa dac trung hanh vi bang StandardScaler...')
    # Danh sach cac cot dac trung duoc lua chon de dua vao phan cum
    cluster_features = RATIO_CATEGORIES + [
        'Credit_Utilization', 'Payment_Ratio', 'Discretionary_Ratio', 'Redemption_Rate'
    ]
    # Thuc hien chuan hoa du lieu bang Z-score thong qua StandardScaler
    X_scaled, scaler, scaled_df = preprocess_features(df_engineered, cluster_features)

    # -------------------------------------------------------------
    # BUOC 5: PHAN CUM K-MEANS VA CHIEU KHONG GIAN PCA 2D
    # -------------------------------------------------------------
    # In thong bao buoc phan cum K-Means
    print('\n[BUOC 5] Phan cum K-Means va chieu khong gian PCA 2D...')
    # Khoi tao doi tuong phan cum K-Means
    cluster_engine = CustomerClustering(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    # Thuc hien phan cum va gan nhan vao tap du lieu
    cluster_labels = cluster_engine.fit_predict(X_scaled)
    # Luu ket qua phan cum vao cot moi cua DataFrame
    df_engineered['Cluster'] = cluster_labels
    # Danh gia chat luong phan cum bang chi so Silhouette va Davies-Bouldin
    metrics = cluster_engine.evaluate(X_scaled)

    # Chieu tap du lieu xuong khong gian 2 chieu PCA
    pca_2d = cluster_engine.fit_transform_pca(X_scaled)
    # Duong dan luu bieu do phan cum PCA
    pca_file = os.path.join(FIGURES_DIR, 'pca_customer_clusters.png')
    # Ve va luu bieu do phan tan PCA
    plot_pca_clusters(pca_2d, cluster_labels, pca_file)

    # Tong hop bang so lieu thong ke dac trung theo tung cum
    persona_summary = cluster_engine.summarize_personas(df_engineered, cluster_labels)
    # Duong dan luu tep CSV bao cao thong ke cum
    persona_csv_path = os.path.join(REPORTS_DIR, 'cluster_personas_summary.csv')
    # Xuat bang thong ke cum ra tep CSV an toan
    save_dataframe_safely(persona_summary, persona_csv_path)

    # -------------------------------------------------------------
    # BUOC 6: TINH TOAN CHI SO DIEM THUONG VA XUAT BAO CAO TUNG KHACH HANG
    # -------------------------------------------------------------
    # In thong bao buoc tinh toan chi so va xuat bao cao tung khach hang
    print('\n[BUOC 6] Tinh toan chi so diem thuong va tong hop bao cao cho tung khach hang...')
    # Khoi tao bo tinh toan diem va phan tich hanh vi
    scorer = CustomerBehaviorScorer(
        base_multipliers=BASE_REWARD_MULTIPLIERS,
        expense_bonuses=EXPENSE_BONUS_FACTORS,
        risk_thresholds=RISK_THRESHOLDS
    )
    # Tinh toan chi so diem thuong dong cho toan bo tap khach hang
    df_engineered['Dynamic_Reward_Points'] = scorer.calculate_dynamic_reward_points(df_engineered)
    # Huan luyen k-NN de xac dinh lang gieng hanh vi tuong dong trong tung cum
    scorer.fit_knn_by_cluster(df_engineered, cluster_features)

    # Tong hop bao cao phan tich chi tiet cho tung khach hang tren toan bo 50.000 dong
    customer_report_df = scorer.generate_customer_report(df_engineered)
    # Duong dan luu tep CSV bao cao chi tiet tung khach hang
    customer_report_path = os.path.join(REPORTS_DIR, 'customer_level_analysis_report.csv')
    # Xuat toan bo bao cao tung khach hang ra tep CSV (luu truc tiep khong in chi tiet ra man hinh)
    save_dataframe_safely(customer_report_df, customer_report_path)

    # -------------------------------------------------------------
    # BUOC 7: KIEM CHUNG MO HINH MACHINE LEARNING
    # -------------------------------------------------------------
    # In thong bao buoc kiem chung mo hinh
    print('\n[BUOC 7] Kiem chung mo hinh Machine Learning (Linear Regression vs Decision Tree vs Random Forest)...')
    # Danh sach cac dac trung dau vao duoc dung de du bao diem thuong
    feature_predictors = [
        'Monthly_Spending', 'Annual_Income', 'Credit_Limit', 'Mobile_App_Login',
        'Payment_Ratio', 'Discretionary_Spending', 'Essential_Spending'
    ]
    # Chuan hoa tap bien doc lap bang StandardScaler
    X_val = StandardScaler().fit_transform(df_engineered[feature_predictors].values)
    # Lay mang gia tri bien phu thuoc muc tieu
    y_val = df_engineered['Dynamic_Reward_Points'].values

    # Khoi tao dong co kiem chung mo hinh Machine Learning
    val_engine = RewardValidationEngine(random_state=RANDOM_STATE)
    # Chay doi chuan hieu nang tren 3 mo hinh
    benchmark_df = val_engine.run_benchmark(X_val, y_val)

    # Duong dan luu bao cao doi chuan dang CSV
    benchmark_csv_path = os.path.join(REPORTS_DIR, 'model_benchmark_metrics.csv')
    # Xuat bang ket qua doi chuan ra CSV an toan
    save_dataframe_safely(benchmark_df, benchmark_csv_path)

    # Duong dan luu bieu do so sanh hieu nang
    benchmark_img_path = os.path.join(FIGURES_DIR, 'model_benchmark_comparison.png')
    # Ve va luu bieu do cot so sanh R2
    plot_model_benchmark(benchmark_df, benchmark_img_path)

    # In thong bao hoan tat toan bo quy trinh
    print('\n' + '=' * 80)
    # In dong tong ket
    print('  QUY TRINH PHAN TICH DU LIEU VA MO HINH HOA DA HOAN TAT XUAT SAC!')
    # In thu muc luu tru bieu do
    print(f'  - Bieu do truc quan hoa da duoc luu tai: {FIGURES_DIR}')
    # In thu muc luu tru bao cao du lieu
    print(f'  - Toan bo bao cao CSV da duoc luu tai  : {REPORTS_DIR}')
    # In duong ke ket thuc
    print('=' * 80)

# Diem bat dau thuc thi chuong trinh
if __name__ == '__main__':
    # Goi ham main de bat dau xu ly
    main()

