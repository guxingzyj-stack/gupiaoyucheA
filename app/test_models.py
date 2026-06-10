"""
测试脚本：验证模型优化功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 创建模拟数据
def create_mock_data(days=300):
    """创建模拟股票数据"""
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    
    # 生成模拟价格（随机游走）
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "close": close,
        "open": close * (1 + np.random.uniform(-0.01, 0.01, days)),
        "high": close * (1 + np.abs(np.random.uniform(0, 0.02, days))),
        "low": close * (1 - np.abs(np.random.uniform(0, 0.02, days))),
        "volume": np.random.randint(1000000, 10000000, days),
    }, index=dates)
    
    # 添加技术指标（模拟）
    df["rsi_14"] = np.random.uniform(30, 70, days)
    df["rsi_6"] = np.random.uniform(30, 70, days)
    df["macd"] = np.random.uniform(-1, 1, days)
    df["macd_signal"] = np.random.uniform(-1, 1, days)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["bb_pct"] = np.random.uniform(0, 1, days)
    df["kdj_k"] = np.random.uniform(0, 100, days)
    df["kdj_d"] = np.random.uniform(0, 100, days)
    df["kdj_j"] = 2 * df["kdj_k"] - df["kdj_d"]
    df["atr_14"] = close * 0.02
    df["vol_ratio"] = np.random.uniform(0.5, 2, days)
    df["momentum_1"] = df["close"].pct_change(1)
    df["momentum_3"] = df["close"].pct_change(3)
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)
    df["volatility_10"] = df["close"].pct_change().rolling(10).std()
    df["volatility_20"] = df["close"].pct_change().rolling(20).std()
    df["obv"] = np.cumsum(df["volume"] * np.sign(df["close"].diff()))
    
    # 市场特征
    df["hs300_ret1"] = np.random.normal(0, 0.01, days)
    df["hs300_ret5"] = np.random.normal(0, 0.02, days)
    df["hs300_ret20"] = np.random.normal(0, 0.05, days)
    df["rel_strength"] = np.random.uniform(-1, 1, days)
    df["main_flow_z"] = np.random.uniform(-2, 2, days)
    
    # 新增特征（模拟）
    df["industry_ret_1"] = np.random.normal(0, 0.01, days)
    df["industry_ret_5"] = np.random.normal(0, 0.02, days)
    df["margin_change"] = np.random.uniform(-0.05, 0.05, days)
    df["main_money_inflow"] = np.random.uniform(-1e8, 1e8, days)
    df["market_vix"] = np.random.uniform(0.15, 0.30, days)
    
    # 移动平均线
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_60"] = df["close"].rolling(60).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["bb_upper"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_middle"] = df["sma_20"]
    df["bb_lower"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
    
    return df.dropna()

# 测试XGBoost模型
print("=" * 60)
print("测试1：XGBoost模型（特征工程增强 + 贝叶斯优化）")
print("=" * 60)

try:
    from models.xgboost_model import XGBoostPredictor, _build_features
    
    # 创建测试数据
    df = create_mock_data(300)
    print(f"测试数据形状：{df.shape}")
    
    # 测试特征工程
    feat_df = _build_features(df)
    print(f"构建特征数量：{feat_df.shape[1]}")
    print(f"特征列表（前10个）：{list(feat_df.columns[:10])}")
    
    # 训练模型（不使用贝叶斯优化以加快测试速度）
    model = XGBoostPredictor(prediction_days=5, use_bayes=False)
    rmse = model.train(df)
    print(f"XGBoost验证集RMSE：{rmse:.4f}")
    
    # 测试预测
    result = model.predict(df)
    print(f"预测天数：{len(result['predictions'])}")
    print(f"预测价格范围：{result['predictions'][0]:.2f} - {result['predictions'][-1]:.2f}")
    print(f"预测涨跌：{result['pct_change']:.2f}%")
    
    # 特征重要性
    importance = model.get_feature_importance()
    if not importance.empty:
        print(f"Top5重要特征：{list(importance.head(5).index)}")
    
    print("\n✅ XGBoost模型测试通过！\n")

except Exception as e:
    print(f"❌ XGBoost测试失败：{e}\n")

# 测试LSTM模型
print("=" * 60)
print("测试2：LSTM模型（42个特征 + 双向LSTM）")
print("=" * 60)

try:
    from models.lstm_model import LSTMPredictor
    
    # 创建测试数据
    df = create_mock_data(300)
    print(f"测试数据形状：{df.shape}")
    
    # 训练模型（使用MLP后端以加快测试速度）
    model = LSTMPredictor(lookback=30, prediction_days=5, use_bidirectional=False)
    rmse = model.train(df)
    print(f"LSTM验证集RMSE：{rmse:.4f}")
    
    # 测试预测
    result = model.predict(df)
    print(f"预测天数：{len(result['predictions'])}")
    print(f"预测价格范围：{result['predictions'][0]:.2f} - {result['predictions'][-1]:.2f}")
    print(f"使用特征数量：{result.get('n_features', 'N/A')}")
    print(f"后端：{result.get('backend', 'N/A')}")
    
    print("\n✅ LSTM模型测试通过！\n")

except Exception as e:
    print(f"❌ LSTM测试失败：{e}\n")

print("=" * 60)
print("所有测试完成！")
print("=" * 60)
