# ===============================================
# 第三步：全局敏感性分析 (SHAP) - 筛选参数；必须运行
# ===============================================
import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt
from data_loader import DataLoader

# 配置 Matplotlib字体以支持中文显示 (如果需要)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SensitivityGSA:
    def __init__(self, model_dir='surrogate-models_opt-results', output_dir='analysis_results'):
        """
        初始化全局敏感性分析器
        :param model_dir: 存放 .pkl 模型文件的目录
        :param output_dir: 分析结果图表和数据的保存目录
        """
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
                raise FileNotFoundError(f"未找到模型文件: {path}，请先运行 surrogate_model.py")

    def run_analysis(self, X_full):
        """
        执行核心 SHAP 分析
        :param X_full: 全量输入特征数据 (DataFrame)
        """
        print("\n" + "="*30 + " 开始全局敏感性分析 (SHAP) " + "="*30)
        
        importance_summary = {} # 用于存储每个目标的特征重要性排序

        for target in self.target_names:
            print(f"\n正在分析目标: [{target}] ...")
            model = self.models[target]
            
            # 1. 初始化 TreeExplainer
            # model 是 XGBRegressor 对象，shap 能够直接识别
            explainer = shap.TreeExplainer(model)
            
            # 2. 计算 SHAP 值
            # 注意：这里计算的是对数空间输出 (ln(y+1)) 的贡献值
            shap_values = explainer.shap_values(X_full) # shap_values[k, i]代表了在第k个样本中，第i个参数/特征（比如 btf）把损伤值的对数预测结果推高或拉低了多少（相对损伤的平均基准值而言）
            
            # 3. 生成并保存 Beeswarm Summary Plot (蜂群图)
            # 这张图展示了：特征的重要性排序 + 特征值大小对输出方向的影响
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_full, show=False, plot_type="dot")
            plt.title(f'{target} SHAP Summary (Log-Scale Impact)', fontsize=14)
            save_path_dot = os.path.join(self.output_dir, f'shap_summary_{target}_dot.png')
            plt.savefig(save_path_dot, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  > 蜂群图已保存: {save_path_dot}")
            
            # 4. 生成并保存 Bar Plot (条形图)
            # 这张图展示了：全局特征重要性 (平均绝对 SHAP 值)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_full, show=False, plot_type="bar")
            plt.title(f'{target} Global Feature Importance', fontsize=14)
            save_path_bar = os.path.join(self.output_dir, f'shap_importance_{target}_bar.png')
            plt.savefig(save_path_bar, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"  > 重要性条形图已保存: {save_path_bar}")

            # 5. 量化特征重要性数据
            # 计算 mean(|SHAP value|)
            mean_abs_shap = np.abs(shap_values).mean(axis=0) # 即特征i的全局重要性I_i: 定义为所有样本绝对贡献的平均值.即特征 i 的变化平均会引起损伤值的对数预测结果变化多少
            feature_importance = pd.DataFrame({
                'Feature': X_full.columns,
                'Importance': mean_abs_shap
            }).sort_values(by='Importance', ascending=False)
            
            # 计算相对占比 (%)
            total_importance = feature_importance['Importance'].sum()
            feature_importance['Relative_Pct'] = (feature_importance['Importance'] / total_importance) * 100
            
            importance_summary[target] = feature_importance
            
            # 保存该目标的详细数据
            csv_path = os.path.join(self.output_dir, f'importance_data_{target}.csv')
            feature_importance.to_csv(csv_path, index=False)
            print(f"  > 详细数据已保存: {csv_path}")

        return importance_summary

    def generate_aggregated_report(self, importance_summary):
        """
        生成综合筛选建议报告
        找出在三个损伤指标中都不重要的参数
        """
        print("\n" + "-"*20 + " 生成综合筛选报告 " + "-"*20)
        
        # 提取所有特征
        all_features = importance_summary['HIC15']['Feature'].tolist()
        
        # 创建一个汇总表
        report = pd.DataFrame({'Feature': all_features})
        
        # 标记是否在某个指标中重要 (例如：相对贡献 > 1% 或 0.5%)
        threshold_pct = 1.0 

        
        combined_importance = pd.DataFrame({'Feature': all_features}).set_index('Feature')
        combined_importance['Total_Score'] = 0.0

        for target, df in importance_summary.items():
            df_indexed = df.set_index('Feature')
            combined_importance[f'{target}_Pct'] = df_indexed['Relative_Pct']
            # 累加重要性分数
            combined_importance['Total_Score'] += df_indexed['Importance']
            
            # 打印 Top 5
            print(f"\n[{target}] Top 5 关键参数:")
            print(df.head(5)[['Feature', 'Relative_Pct']].to_string(index=False))

        # 排序
        combined_importance = combined_importance.sort_values(by='Total_Score', ascending=False)
        
        # 保存汇总表
        summary_path = os.path.join(self.output_dir, 'FINAL_feature_ranking_summary.csv')
        combined_importance.to_csv(summary_path)
        print(f"\n汇总排序表已保存: {summary_path}")
        
        # # 识别建议剔除/固定的参数 (Bottom Parameter)
        # # 逻辑：在三个指标中的相对占比之和都很低
        # print("\n=== 新版采样策略建议 ===")
        # print("建议固定或大幅缩小采样范围的参数 (综合重要性极低):")
        # tail_features = combined_importance.tail(5) # 看最后5个
        # print(tail_features)

if __name__ == "__main__":
    # 1. 准备全量数据
    data_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0408_del.csv'
    
    if os.path.exists(data_path):
        loader = DataLoader(data_path)
        # 这里只需要 X, y，不需要切分，因为我们做分析用全量数据
        X_train, X_test, y_train, y_test = loader.load_data()
        
        # 合并全量数据
        X_full = pd.concat([X_train, X_test])
        
        # 2. 运行 GSA 分析
        # 确保 model_dir 与 surrogate_model.py 中保存的路径一致
        gsa = SensitivityGSA(model_dir='surrogate-models_opt-results-mae-PS', output_dir='analysis_results_gsa')
        gsa.load_models()
        
        summary = gsa.run_analysis(X_full)
        gsa.generate_aggregated_report(summary)
        
        print("\n敏感性分析全部完成。请查看 'analysis_results_gsa' 文件夹。")
        
    else:
        print(f"错误: 找不到数据文件 {data_path}")