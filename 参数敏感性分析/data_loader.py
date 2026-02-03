# ===============================================
# 第一步：数据加载、清洗、特征解耦、划分数据集; 不需要手动运行
# ===============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath, test_size=0.15, random_state=42):
        """
        初始化数据加载器
        :param filepath: distribution 文件路径 (.csv 或 .npz)
        :param test_size: 测试集比例
        :param random_state: 随机种子，保证划分的可复现性
        """
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = None
        self.target_names = ['HIC15', 'Dmax', 'Nij']
        
        # 剔除除了特征和标签外的无关列
        # 除了is_driver_side外，剩余的有效输入特征共11个
        self.drop_columns = [
            'is_driver_side',
            'DZ', 'PTF', # DZ 和 OT 一一对应，冗余; PTF 与 BTF 完全线性相关，冗余
            'case_id', 'have_run', 'is_pulse_ok', 'is_injury_ok',
            'delta_vx(kph)', 'delta_vy(kph)', 'delta_v(kph)',
        ]

    def load_data(self):
        """
        加载并预处理数据
        :return: X_train, X_test, y_train, y_test (DataFrames)
        """
        print(f"正在加载数据: {self.filepath}")
        
        # 1. 读取数据
        if self.filepath.endswith('.csv'):
            df = pd.read_csv(self.filepath)
        elif self.filepath.endswith('.npz'):
            data = np.load(self.filepath, allow_pickle=True)
            df = pd.DataFrame({key: data[key] for key in data.files})
        else:
            raise ValueError("不支持的文件格式，仅支持 .csv 或 .npz")

        # 2. 数据有效性过滤 (Strict Filtering)
        # 必须确保仿真成功、脉冲波形有效、且损伤计算有效
        initial_count = len(df)
        df = df[
            (df['have_run'] == True) & 
            (df['is_pulse_ok'] == True) & 
            (df['is_injury_ok'] == True)
        ].copy()
        
        print(f"数据过滤完成: {initial_count} -> {len(df)} (剔除无效样本)")
        
        # 3. 特征解耦与清洗
        # 移除不需要的列
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop)
        
        # 4. 分离特征 (X) 和 目标 (Y)
        # 确保目标列存在
        missing_targets = [t for t in self.target_names if t not in df_clean.columns]
        if missing_targets:
            raise ValueError(f"数据中缺少目标列: {missing_targets}")
            
        y = df_clean[self.target_names]
        X = df_clean.drop(columns=self.target_names)
        
        self.feature_names = X.columns.tolist()
        print(f"输入特征 ($X$): {self.feature_names}")
        print(f"特征维度: {X.shape[1]}")

        # 5. 数据集划分
        # 使用 stratify=None，因为这是回归问题。
        # 如果需要分层，通常按 'occupant_type' 分层比较合理，但在大样本下随机划分通常足够。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"数据集划分完成: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")
        
        return X_train, X_test, y_train, y_test

# --- 单元测试部分 (实际运行时可注释) ---
if __name__ == "__main__":
    # 替换为你实际的文件路径进行测试
    data_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0121.csv' 
    try:
        loader = DataLoader(data_path)
        X_tr, X_te, y_tr, y_te = loader.load_data()
        print("\n数据加载测试通过。")
        print("X_train head:\n", X_tr.head())
        print("y_train head:\n", y_tr.head())
    except Exception as e:
        print(f"测试失败: {e}")