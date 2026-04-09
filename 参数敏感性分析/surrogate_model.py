# ===============================================
# 第二步：构建并训练 XGBoost 代理模型，评估拟合精度；必须运行
# ===============================================
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from data_loader import DataLoader

# 设置 Optuna 的日志级别，避免刷屏
optuna.logging.set_verbosity(optuna.logging.WARNING)

class SurrogateModel:
    def __init__(self, target_names=['HIC15', 'Dmax', 'Nij'], n_trials=100, storage_dir='optuna_results'):
        """
        初始化代理模型管理器 (支持自动超参优化)
        :param target_names: 需要预测的目标变量列表
        :param n_trials: Optuna 尝试寻找最优超参的次数，次数越多越精准但越慢
        :param storage_dir: Optuna 结果存储目录
        """
        self.target_names = target_names
        self.n_trials = n_trials
        self.models = {}
        self.best_params = {}
        self.studies = {}  # 存储 study 对象用于后续可视化
        
        # 设置存储路径
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.db_path = os.path.join(storage_dir, 'optuna_studies.db')
        self.storage_url = f"sqlite:///{self.db_path}"

        # 定义单调性约束
        # 1: 正相关 (随着特征值增加，目标值增加)
        # 0: 无约束
        # -1: 负相关
        self.monotone_constraints = {"impact_velocity": 1}
        
        # 打印 Dashboard 启动命令
        print(f"="*60)
        print(f"💡 实时查看寻优过程,请在另一个终端运行:")
        # 将 Windows 反斜杠转换为正斜杠用于 SQLite URL
        db_path_url = self.db_path.replace('\\', '/')
        print(f"   optuna-dashboard sqlite:///{db_path_url}")
        print(f"="*60 + "\n")
        
        # 默认备用参数（万一优化失败使用）
        self.default_params = {
            'n_estimators': 2000,
            'learning_rate': 0.015,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'reg_lambda': 1e-4,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'random_state': 2025
        }

    def _transform_target(self, y):
        """对目标变量进行 Log 变换: ln(y + 1)"""
        return np.log1p(y)

    def _inverse_transform_target(self, y_log):
        """对预测结果进行反变换: exp(y) - 1"""
        pred = np.expm1(y_log)
        return np.maximum(pred, 0)

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna 的目标函数：定义超参搜索空间并返回验证集误差
        """
        # 定义搜索空间
        params = {
            # 'n_estimators': trial.suggest_int('n_estimators', 1000, 2400, step=100),# 1000 到 2400 之间选择 #
            'n_estimators': trial.suggest_int('n_estimators', 1800, 2200, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 6, step=1), #
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),#
            # 'subsample': trial.suggest_float('subsample', 0.7, 0.9),# 
            'subsample': trial.suggest_float('subsample', 0.7, 0.85, step=0.01), #
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.01), #
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-2, log=True), # L2 正则 #
            # 固定参数
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'random_state': 2025,
            # 注入单调性约束
            'monotone_constraints': self.monotone_constraints
        }
        
        # 训练临时模型
        model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        # ==================================================================================
        # ========== 这里可以选择不同的评估指标进行优化 ==========
        # ==================================================================================
        # rmse = np.sqrt(mean_squared_error(y_val, preds))
        # return rmse
        mae = mean_absolute_error(y_val, preds)
        return mae
        # ==================================================================================
        # ==================================================================================


    def optimize_hyperparameters(self, X_train_full, y_train_full, target_name):
        """
        核心功能：自动为指定目标寻找最优超参
        """
        print(f"  > [{target_name}] 正在进行超参优化 (Optuna {self.n_trials} trials)...")
        
        # 1. 从训练集中再划分出一个验证集用于调参 (不触碰外部的 X_test)
        # 比例 8:2
        X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
            X_train_full, y_train_full, test_size=0.16, random_state=2025
        )
        
        # 2. 转换目标变量 (Log)
        y_opt_train_log = self._transform_target(y_opt_train)
        y_opt_val_log = self._transform_target(y_opt_val)
        
        # 使用 SQLite 存储，支持实时 Dashboard 查看
        study = optuna.create_study(
            study_name=f"XGB_{target_name}",
            storage=self.storage_url,
            direction='minimize',
            load_if_exists=True  # 支持断点续训
        )
        study.optimize(
            lambda trial: self._objective(trial, X_opt_train, y_opt_train_log, X_opt_val, y_opt_val_log),
            n_trials=self.n_trials,
            show_progress_bar=True  # 显示进度条
        )
        
        self.studies[target_name] = study  # 保存用于可视化
        
        print(f"  > [{target_name}] 最优参数找到! Val metric: {study.best_value:.4f}")
        # print(f"    Params: {study.best_params}")
        
        # 补充固定参数并保存
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'random_state': 2025,
            # 确保最优参数中包含约束，供后续全量训练使用
            'monotone_constraints': self.monotone_constraints
        })
        
        self.best_params[target_name] = best_params
        return best_params

    def plot_optimization_results(self):
        """生成并保存所有目标的优化可视化图"""
        print("\n生成优化过程可视化...")
        
        for target, study in self.studies.items():
            # 优化历史图
            fig1 = plot_optimization_history(study)
            fig1.write_html(os.path.join(self.storage_dir, f'{target}_history.html'))
            
            # 参数重要性图
            if len(study.trials) > 1:
                fig2 = plot_param_importances(study)
                fig2.write_html(os.path.join(self.storage_dir, f'{target}_importance.html'))
        
        print(f"可视化结果已保存至: {self.storage_dir}/")

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        阶段一：自动调参 -> 确定最佳迭代次数 -> 最终评估
        """
        print("="*30 + " 阶段一：自动调参、训练与评估 " + "="*30)
        metrics_summary = {}

        for target in self.target_names:
            print(f"\n处理目标: [{target}]")
            
            # 1. 自动寻找学习率、树深等参数
            best_params = self.optimize_hyperparameters(X_train, y_train[target], target)
            
            # 2. 关键修正步骤：确定最佳迭代次数 (n_estimators)
            # 必须从 X_train 中分离出一个验证集，专门用来确定树的数量，绝不能看 X_test
            print(f"  > 使用训练集内部划分验证，确定最佳迭代次数...")
            X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
                X_train, y_train[target], test_size=0.16, random_state=2025
            )
            
            y_tr_log = self._transform_target(y_tr_inner)
            y_val_log = self._transform_target(y_val_inner)
            
            # 临时模型：只为了找 best_iteration
            temp_model = xgb.XGBRegressor(**best_params, early_stopping_rounds=100)
            temp_model.fit(
                X_tr_inner, y_tr_log,
                eval_set=[(X_val_inner, y_val_log)],
                verbose=False
            )
            
            # 获取早停后的最佳树数量
            best_num_trees = temp_model.best_iteration
            print(f"  > 监测到最佳树数量 (Best Iteration): {best_num_trees}(全量训练时比这个再大10%), 原设定 n_estimators={best_params['n_estimators']}")
            
            # ‼️ 更新 best_params，将 n_estimators 锁定为最佳值
            # 这样后续的全量训练就不会跑满 n_estimators最大值（如2400） 次了
            self.best_params[target]['n_estimators'] = int(best_num_trees*1.1)  # 稍微放宽一点，防止欠拟合
            
            # 3. 使用锁定的最佳树数量，在完整的 X_train 上重训 (Refit)，用于评估
            # 此时不需要早停了，因为 n_estimators 已经是修正过的最佳值
            final_eval_model = xgb.XGBRegressor(**self.best_params[target])
            
            y_train_full_log = self._transform_target(y_train[target])
            final_eval_model.fit(X_train, y_train_full_log, verbose=False)
            
            # 4. 最终评估 (此时 X_test 才是真正纯净的)
            pred_log = final_eval_model.predict(X_test)
            pred_original = self._inverse_transform_target(pred_log)
            y_test_original = y_test[target].values
            
            r2 = r2_score(y_test_original, pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
            mae = mean_absolute_error(y_test_original, pred_original)
            
            metrics_summary[target] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            print(f"  > [{target}] 最终测试集评估: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

        return metrics_summary

    def fit_full_dataset(self, X_full, y_full):
        """
        阶段二：使用最优参数（含最佳迭代次数）全量重训
        """
        print("\n" + "="*30 + " 阶段二：全量数据重训 (Full Retrain) " + "="*30)
        
        for target in self.target_names:
            # 1. 获取该目标的最优参数 
            # (注意：此时 params 里的 n_estimators 已经被 update 为 best_num_trees 了)
            params = self.best_params.get(target, self.default_params)
            print(f"正在全量重训目标: [{target}] 使用参数: n_estimators={params.get('n_estimators')} ...")
            
            # 2. 数据预处理
            y_full_log = self._transform_target(y_full[target])
            
            # 3. 训练 (不需早停，直接跑完固定的 n_estimators)
            final_params = params.copy()
            
            model = xgb.XGBRegressor(**final_params)
            model.fit(X_full, y_full_log, verbose=False)
            
            self.models[target] = model
            
        print("全量模型训练完成。")

    def save_models(self, save_dir='models'):
        """保存模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for target, model in self.models.items():
            path = os.path.join(save_dir, f'xgb_{target}.pkl')
            joblib.dump(model, path)
            print(f"模型已保存: {path}")

    def save_training_results(self, metrics_summary, save_dir='models'):
        """保存最优超参数和评估结果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存最优超参数
        params_df = pd.DataFrame(self.best_params).T
        params_path = os.path.join(save_dir, 'best_hyperparameters.csv')
        params_df.to_csv(params_path, encoding='utf-8-sig')
        print(f"最优超参数已保存: {params_path}")
        
        # 保存评估指标
        metrics_df = pd.DataFrame(metrics_summary).T
        metrics_path = os.path.join(save_dir, 'test_metrics.csv')
        metrics_df.to_csv(metrics_path, encoding='utf-8-sig')
        print(f"测试集评估结果已保存: {metrics_path}")
        
        # 保存综合报告 (可读性更好的文本格式)
        report_path = os.path.join(save_dir, 'training_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("XGBoost 代理模型训练报告\n")
            f.write("="*60 + "\n\n")
            
            for target in self.target_names:
                f.write(f"\n{'='*20} {target} {'='*20}\n\n")
                
                # 最优超参数
                f.write("最优超参数:\n")
                for param, value in self.best_params[target].items():
                    f.write(f"  {param}: {value}\n")
                
                # 评估指标
                f.write(f"\n测试集评估:\n")
                f.write(f"  R²:   {metrics_summary[target]['R2']:.4f}\n")
                f.write(f"  RMSE: {metrics_summary[target]['RMSE']:.2f}\n")
                f.write(f"  MAE:  {metrics_summary[target]['MAE']:.2f}\n")
        
        print(f"训练报告已保存: {report_path}")

    def load_models(self, save_dir='models'):
        """加载模型"""
        for target in self.target_names:
            path = os.path.join(save_dir, f'xgb_{target}.pkl')
            if os.path.exists(path):
                self.models[target] = joblib.load(path)
                print(f"模型已加载: {path}")
            else:
                print(f"警告: 找不到模型文件 {path}")
                
    def predict(self, X):
        """预测接口"""
        results = {}
        for target in self.target_names:
            if target not in self.models:
                raise ValueError(f"模型 {target} 尚未训练")
            pred_log = self.models[target].predict(X)
            results[target] = self._inverse_transform_target(pred_log)
        return pd.DataFrame(results, index=X.index if hasattr(X, 'index') else None)

if __name__ == "__main__":
    # 1. 准备数据
    # 请确保 data_loader.py 在同一目录下，且路径正确
    data_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0408_del.csv'
    
    if os.path.exists(data_path):
        # 加载数据
        loader = DataLoader(data_path)
        X_train, X_test, y_train, y_test = loader.load_data()
        save_dir = 'surrogate-models_opt-results-mae'

        # 2. 初始化全自动模型
        surrogate = SurrogateModel(target_names=['HIC15', 'Dmax', 'Nij'], n_trials=100, storage_dir=save_dir)
        
        # 3. 自动调参 + 验证评估
        # 这步会自动打印每个目标的优化过程和最终 R2
        metrics = surrogate.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # 4. 自动全量重训
        # 合并数据
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        surrogate.fit_full_dataset(X_full, y_full)
        
        # 5. 自动保存模型和训练结果
        surrogate.save_models(save_dir=save_dir)
        surrogate.save_training_results(metrics, save_dir=save_dir)
        
        # # 6. 生成可视化
        # surrogate.plot_optimization_results()
        
        print("\n所有步骤自动完成！")
        print(f"  - 模型保存在: '{save_dir}'")
        print(f"  - 训练结果保存在: '{save_dir}' (包含超参数和评估指标)")
        # print(f"  - 寻优历史保存在: '{surrogate.storage_dir}/' (可用浏览器打开 .html 文件)")
        
    else:
        print(f"错误: 找不到文件 {data_path}")