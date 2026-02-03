# ==================================================
# 第四步：局部敏感性分析 (ALE) - 识别关键区间；必须运行，可根据第三步结果调整分析特征
# ==================================================
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from PyALE import ale
from data_loader import DataLoader

# 配置绘图风格和字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False

class ModelWrapper:
    """
    模型包装器
    用于欺骗 PyALE，让它以为模型直接输出的是原始物理量纲 (HIC, mm 等)
    而不是 Log 值。
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        # 1. 调用原始 XGBoost 预测 (得到 Log 值)
        pred_log = self.model.predict(X)
        # 2. 实时还原为物理值: exp(y) - 1
        # 使用 np.maximum 截断负值，保证物理合理性
        return np.maximum(np.expm1(pred_log), 0)

class SensitivityLSA:
    def __init__(self, model_dir='surrogate-models_opt-results', output_dir='analysis_results_lsa'):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.target_names = ['HIC15', 'Dmax', 'Nij']
        self.models = {}
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_models(self):
        """加载训练好的 XGBoost 模型"""
        print(f"从 {self.model_dir} 加载模型...")
        for target in self.target_names:
            path = os.path.join(self.model_dir, f'xgb_{target}.pkl')
            if os.path.exists(path):
                self.models[target] = joblib.load(path)
            else:
                raise FileNotFoundError(f"未找到模型: {path}")

    def run_1d_analysis(self, X_full, features_to_analyze):
        """生成 1D ALE 趋势图"""
        print("\n" + "="*30 + " 开始局部敏感性分析 (1D ALE) " + "="*30)
        
        for target in self.target_names:
            print(f"\n正在分析目标: [{target}] 的局部趋势 (原始物理量纲)...")
            
            # 使用包装器包裹模型，确保 ALE 图显示的是真实物理值
            original_model = self.models[target]
            wrapped_model = ModelWrapper(original_model)
            
            for feature in features_to_analyze:
                if feature not in X_full.columns:
                    continue

                print(f"  > 绘制特征 [{feature}] 的 ALE 曲线...")
                
                try:
                    # 调用 PyALE
                    ale_eff = ale(
                        X=X_full, 
                        model=wrapped_model,  # 传入包装后的模型
                        feature=[feature], 
                        grid_size=45, 
                        include_CI=True,
                        plot=False 
                    )
                    
                    self._custom_plot_1d(ale_eff, feature, target)
                    
                except Exception as e:
                    print(f"    ! 警告: 分析特征 {feature} 时出错: {e}")

    def _custom_plot_1d(self, ale_eff, feature, target):
        """自定义绘制 1D ALE 图"""
        df_ale = ale_eff 
        
        plt.figure(figsize=(8, 6))
        
        # 绘制 ALE 曲线
        plt.plot(df_ale.index, df_ale['eff'], color='darkblue', linewidth=2, label='Effect')
        
        # 绘制置信区间
        if 'lowerCI_95%' in df_ale.columns:
            plt.fill_between(
                df_ale.index, 
                df_ale['lowerCI_95%'], 
                df_ale['upperCI_95%'], 
                color='lightblue', alpha=0.4, label='95% CI'
            )
            
        plt.title(f'1D Trend: {feature} -> {target}', fontsize=14)
        plt.xlabel(f'{feature} (Original Unit)', fontsize=12)
        plt.ylabel(f'Effect on {target} (Change in Value)', fontsize=12)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, f'ALE_1D_{target}_{feature}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def run_2d_interaction(self, X_full, feature_pair):
        """生成 2D ALE 交互热力图"""
        f1, f2 = feature_pair
        print(f"\n正在分析 2D 交互作用: {f1} vs {f2} ...")
        
        if f1 not in X_full.columns or f2 not in X_full.columns:
            print("  ! 特征不存在，跳过 2D 分析")
            return

        for target in self.target_names:
            # 同样使用包装器
            original_model = self.models[target]
            wrapped_model = ModelWrapper(original_model)
            
            try:
                # 2D ALE 计算
                ale_eff_2d = ale(
                    X=X_full, 
                    model=wrapped_model, 
                    feature=[f1, f2], 
                    grid_size=50,
                    plot=False 
                )
                
                # PyALE 2D 返回的 DataFrame 列通常是 [feature1, feature2, eff]
                # Pivot 数据以用于热力图
                eff_matrix = ale_eff_2d.pivot(index=f2, columns=f1, values='eff')
                
                plt.figure(figsize=(10, 8))
                plt.imshow(
                    eff_matrix, 
                    aspect='auto', 
                    origin='lower',
                    extent=[eff_matrix.columns.min(), eff_matrix.columns.max(), 
                            eff_matrix.index.min(), eff_matrix.index.max()],
                    cmap='RdBu_r' 
                )
                plt.colorbar(label=f'Effect on {target}')
                plt.title(f'2D Interaction: {f1} & {f2} -> {target}')
                plt.xlabel(f1)
                plt.ylabel(f2)
                plt.grid(False) # 热力图不需要网格
                
                save_path = os.path.join(self.output_dir, f'ALE_2D_{target}_{f1}_vs_{f2}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"  > [{target}] 2D 热力图已保存: {save_path}")
                
            except Exception as e:
                # 打印详细错误以便调试
                import traceback
                print(f"  ! [{target}] 2D 分析失败: {e}")
                # traceback.print_exc()

if __name__ == "__main__":
    # 1. 准备全量数据
    data_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0121.csv'
    
    if os.path.exists(data_path):
        loader = DataLoader(data_path)
        X_train, X_test, y_train, y_test = loader.load_data()
        
        # reset_index(drop=True) 是必须的,否则 PyALE 在处理 2D 交互时会因为索引不连续而抛出 KeyError
        X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
        
        print(f"全量数据构建完成，Shape: {X_full.shape}")
        
        # 2. 初始化 LSA 分析器
        lsa = SensitivityLSA(model_dir='surrogate-models_opt-results-mae', output_dir='analysis_results_lsa')
        lsa.load_models()
        
        # 3. 定义需要重点分析的参数
        top_features = [
            'impact_velocity',
            'impact_angle',
            'overlap',
            'OT', 
            'LL1', 
            'LL2', 
            'BTF',  
            'LLATTF',          
            'AFT', 
            'SP', 
            'RA', 
            # 'is_driver_side'
        ]
        
        # 4. 执行 1D 趋势分析
        lsa.run_1d_analysis(X_full, top_features)
        
        # # 5. 执行 2D 交互分析
        # lsa.run_2d_interaction(X_full, ['impact_velocity', 'BTF'])
        # lsa.run_2d_interaction(X_full, ['LL1', 'LL2'])
        # lsa.run_2d_interaction(X_full, ['OT', 'LLATTF'])
        # ...
        
        print("\n局部敏感性分析完成。请查看 'analysis_results_lsa' 文件夹。")
        
    else:
        print(f"错误: 找不到数据文件 {data_path}")