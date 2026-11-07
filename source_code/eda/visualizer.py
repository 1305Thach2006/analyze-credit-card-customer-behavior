# Import thu vien matplotlib de ve do thi khoa hoc
import matplotlib.pyplot as plt
# Import thu vien seaborn de truc quan hoa thong ke
import seaborn as sns
# Import thu vien pandas de thao tac du lieu
import pandas as pd
# Import thu vien numpy de tinh toan ma tran
import numpy as np

# Cai dat phong cach mac dinh cho cac bieu do truc quan hoa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Ham ve va xuat bieu do ma tran nhiet he so tuong quan
def plot_correlation_heatmap(df: pd.DataFrame, columns: list, output_path: str) -> None:
    # Tao khung hinh ve voi kich thuoc 10x8 inch
    plt.figure(figsize=(10, 8))
    # Tinh toan ma tran he so tuong quan tuyen tinh Pearson
    corr_matrix = df[columns].corr()
    # Ve bieu do nhiet voi bang mau coolwarm va hien thi he so lam tron 2 chu so
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar=True)
    # Dat tieu de cho bieu do tuong quan
    plt.title('Ma tran he so tuong quan giua cac dac trung chi tieu va tai chinh', fontsize=12, fontweight='bold')
    # Can chinh bo cuc khung ve
    plt.tight_layout()
    # Luu hinh anh bieu do ra tap tin voi do phan giai 300 DPI
    plt.savefig(output_path, dpi=300)
    # Dong khung ve de giai phong bo nho
    plt.close()
    # In thong bao da xuat tap tin anh thanh cong
    print(f"[EDA] Da luu bieu do Heatmap tai: {output_path}")

# Ham ve bieu do hop the hien phan phoi chi tieu theo 7 danh muc
def plot_category_spending_boxplot(df: pd.DataFrame, spending_cols: list, output_path: str) -> None:
    # Tao khung hinh ve voi kich thuoc 12x6 inch
    plt.figure(figsize=(12, 6))
    # Chuyen du lieu sang dang dai de ve bieu do hop cho tung danh muc
    melted_df = df[spending_cols].melt(var_name='Danh_Muc', value_name='Chi_Tieu_Thang')
    # Rut gon ten danh muc de truc hoanh hien thi ngan gon
    melted_df['Danh_Muc'] = melted_df['Danh_Muc'].str.replace('_Spending', '')
    # Ve bieu do boxplot the hien trung vi va khoang tu phan vi
    sns.boxplot(x='Danh_Muc', y='Chi_Tieu_Thang', data=melted_df, hue='Danh_Muc', palette='Set2', legend=False)
    # Dat tieu de cho bieu do
    plt.title('Phan phoi chi tieu hang thang theo 7 danh muc tieu dung', fontsize=12, fontweight='bold')
    # Dat nhan truc x
    plt.xlabel('Danh muc chi tieu', fontsize=11)
    # Dat nhan truc y
    plt.ylabel('So tien chi tieu hang thang', fontsize=11)
    # Can chinh bo cuc khung ve
    plt.tight_layout()
    # Luu hinh anh ra tap tin voi do phan giai 300 DPI
    plt.savefig(output_path, dpi=300)
    # Dong khung ve
    plt.close()
    # In thong bao hoan thanh
    print(f"[EDA] Da luu bieu do Boxplot tai: {output_path}")

# Ham ve bieu do phan cum khach hang tren khong gian giam chieu PCA 2D
def plot_pca_clusters(pca_2d: np.ndarray, labels: np.ndarray, output_path: str) -> None:
    # Tao khung hinh ve voi kich thuoc 9x7 inch
    plt.figure(figsize=(9, 7))
    # Dinh nghia danh sach 4 mau sac tuong ung voi 4 cum
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # Dinh nghia ten nhan cum cho tung nhom
    cluster_names = ['Cum 0', 'Cum 1', 'Cum 2', 'Cum 3']
    # Lap qua tung cum de ve cac diem du lieu
    for cluster_id in range(len(colors)):
        # Loc cac diem du lieu thuoc cum hien tai
        cluster_mask = (labels == cluster_id)
        # Ve bieu do phan tan cho cac diem thuoc cum
        plt.scatter(
            pca_2d[cluster_mask, 0],
            pca_2d[cluster_mask, 1],
            c=colors[cluster_id],
            label=cluster_names[cluster_id],
            alpha=0.6,
            edgecolors='w',
            s=35
        )
    # Dat tieu de cho bieu do phan cum
    plt.title('Truc quan hoa cac cum khach hang tren khong gian PCA 2D', fontsize=12, fontweight='bold')
    # Dat nhan cho truc thanh phan chinh thu nhat
    plt.xlabel('Thanh phan chinh 1 (PCA 1)', fontsize=11)
    # Dat nhan cho truc thanh phan chinh thu hai
    plt.ylabel('Thanh phan chinh 2 (PCA 2)', fontsize=11)
    # Hien thi chu thich cac cum
    plt.legend(title='Phan cum', loc='best', frameon=True)
    # Can chinh bo cuc khung ve
    plt.tight_layout()
    # Luu hinh anh ra tap tin
    plt.savefig(output_path, dpi=300)
    # Dong khung ve
    plt.close()
    # In thong bao hoan thanh
    print(f"[EDA] Da luu bieu do PCA Clusters tai: {output_path}")

# Ham ve bieu do cot so sanh do chinh xac R2 giua cac mo hinh du bao
def plot_model_benchmark(benchmark_df: pd.DataFrame, output_path: str) -> None:
    # Tao khung hinh ve voi kich thuoc 9x5 inch
    plt.figure(figsize=(9, 5))
    # Ve bieu do cot the hien he so xac dinh R2 cua cac mo hinh
    bars = plt.bar(benchmark_df['Model'], benchmark_df['R2_Score'], color=['#4a90e2', '#50e3c2', '#f5a623'], width=0.45)
    # Dat tieu de cho bieu do
    plt.title('So sanh he so xac dinh R2 giua cac mo hinh Machine Learning', fontsize=12, fontweight='bold')
    # Dat nhan cho truc tung
    plt.ylabel('He so R2 Score', fontsize=11)
    # Gioi han truc tung tu 0 den 1.1 de hien thi ro gia tri
    plt.ylim(0, 1.1)
    # Hien thi gia tri cu the tren dau tung cot
    for bar in bars:
        # Lay chieu cao cua cot
        yval = bar.get_height()
        # In gia tri len tren dinh cot
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    # Can chinh bo cuc khung ve
    plt.tight_layout()
    # Luu hinh anh bieu do
    plt.savefig(output_path, dpi=300)
    # Dong khung ve
    plt.close()
    # In thong bao hoan thanh
    print(f"[EDA] Da luu bieu do Model Benchmark tai: {output_path}")
