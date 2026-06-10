"""
A股预测模型训练测试脚本
运行完整训练流程并收集RMSE结果
"""
import sys
import os
import pickle
import traceback
import warnings
warnings.filterwarnings("ignore")

os.chdir("C:/A股智能分析预测系统A/app")
sys.path.insert(0, "C:/A股智能分析预测系统A/app")

def load_stock_data():
    """加载股票数据"""
    cache_dir = ".cache"
    files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]

    # 找有完整OHLCV数据的文件
    for f in files:
        path = os.path.join(cache_dir, f)
        with open(path, 'rb') as fp:
            df = pickle.load(fp)
        if isinstance(df, dict):
            df = df.get('data', df.get('df'))
        if df is not None and len(df) > 200:
            # 确保有基本列
            required = ['open', 'close', 'high', 'low', 'volume']
            if all(c in df.columns for c in required):
                return df, f
    return None, None

def build_features(df):
    """构建技术指标"""
    import numpy as np
    import pandas as pd

    # 复制避免修改原数据
    df = df.copy()

    # 基础列名标准化
    rename = {
        '收盘价': 'close', '开盘价': 'open', '最高价': 'high', '最低价': 'low',
        '成交量': 'volume', '成交额': 'amount', '换手率': 'turnover',
        '涨跌幅': 'pct_change', '涨跌额': 'change'
    }
    df.rename(columns=rename, inplace=True)

    # 确保必要列存在
    for col in ['close', 'open', 'high', 'low', 'volume']:
        if col not in df.columns:
            print(f"⚠️ 缺少列: {col}")
            return None

    # 移动平均线
    for w in [5, 10, 20, 60]:
        df[f'sma_{w}'] = df['close'].rolling(w).mean()
        if w in [12, 26]:
            df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()

    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_60'] = df['close'].rolling(60).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    for period in [6, 14]:
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (rs + 1))

    # 布林带
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # KDJ
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    rsv = (df['close'] - low14) / (high14 - low14 + 1e-10) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # 成交量比率
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # 动量
    for d in [1, 3, 5, 10, 20]:
        df[f'momentum_{d}'] = df['close'].pct_change(d)

    # 波动率
    for d in [10, 20]:
        df[f'volatility_{d}'] = df['close'].pct_change().rolling(d).std()

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # 收益率滞后
    for lag in [1, 2, 3, 5, 7, 10, 15, 20]:
        df[f'ret_lag_{lag}'] = df['close'].pct_change(lag)

    # 价格滞后
    for lag in [1, 2, 3, 5, 10]:
        df[f'price_lag_{lag}'] = df['close'].shift(lag)

    # 滚动窗口特征
    for w in [5, 10, 20, 60]:
        df[f'close_ma_ratio_{w}'] = df['close'] / df['close'].rolling(w).mean()
        df[f'vol_ma_ratio_{w}'] = df['volume'] / df['volume'].rolling(w).mean()
        df[f'volatility_{w}'] = df['close'].pct_change().rolling(w).std()
        df[f'high_max_{w}'] = df['high'].rolling(w).max()
        df[f'low_min_{w}'] = df['low'].rolling(w).min()

    # 去除NaN
    df.dropna(inplace=True)

    return df

def test_models(df):
    """测试各模型训练"""
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print("\n" + "="*60)
    print("📊 模型训练测试")
    print("="*60)
    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"特征数量: {len(df.columns)}")

    results = {}

    # 1. XGBoost测试
    print("\n" + "-"*40)
    print("🔍 测试 XGBoost (69特征 + 贝叶斯优化)")
    print("-"*40)
    try:
        from models.xgboost_model import XGBoostPredictor
        import config

        predictor = XGBoostPredictor(prediction_days=config.SHORT_TERM_DAYS, use_bayes=True)
        rmse = predictor.train(df)
        results['xgboost'] = rmse
        print(f"✅ XGBoost 训练完成! 验证RMSE: {rmse:.4f}")
    except Exception as e:
        print(f"❌ XGBoost 训练失败: {e}")
        traceback.print_exc()

    # 2. LSTM测试
    print("\n" + "-"*40)
    print("🔍 测试 LSTM (双向 + Attention)")
    print("-"*40)
    try:
        from models.lstm_model import LSTMPredictor
        import config

        predictor = LSTMPredictor(lookback=20, prediction_days=config.SHORT_TERM_DAYS)
        rmse = predictor.train(df)
        results['lstm'] = rmse
        print(f"✅ LSTM 训练完成! 验证RMSE: {rmse:.4f}")
    except Exception as e:
        print(f"❌ LSTM 训练失败: {e}")
        traceback.print_exc()

    # 3. Ensemble测试
    print("\n" + "-"*40)
    print("🔍 测试 Ensemble (动态权重 + Stacking)")
    print("-"*40)
    try:
        from models.ensemble import EnsemblePredictor
        import config

        predictor = EnsemblePredictor(use_dynamic_weights=True)
        rmse_dict = predictor.train(df)
        results['ensemble'] = rmse_dict
        print(f"✅ Ensemble 训练完成!")
        print("各模型RMSE:")
        for name, rmse in rmse_dict.items():
            print(f"  - {name}: {rmse:.4f}")
    except Exception as e:
        print(f"❌ Ensemble 训练失败: {e}")
        traceback.print_exc()

    return results

def main():
    print("="*60)
    print("🚀 A股预测模型 - 完整训练测试")
    print("="*60)

    # 加载数据
    print("\n📂 加载股票数据...")
    df, filename = load_stock_data()
    if df is None:
        print("❌ 没有找到合适的股票数据")
        return None

    print(f"✅ 加载: {filename}")
    print(f"   数据量: {len(df)} 行")

    # 构建特征
    print("\n🔧 构建技术指标特征...")
    df = build_features(df)
    if df is None:
        print("❌ 特征构建失败")
        return None

    print(f"✅ 特征构建完成: {len(df.columns)} 列")

    # 测试模型
    results = test_models(df)

    # 汇总
    print("\n" + "="*60)
    print("📊 训练结果汇总")
    print("="*60)
    if results:
        print("\n| 模型 | RMSE |")
        print("|------|------|")
        if 'xgboost' in results:
            print(f"| XGBoost | {results['xgboost']:.4f} |")
        if 'lstm' in results:
            print(f"| LSTM | {results['lstm']:.4f} |")
        if 'ensemble' in results:
            print("\n集成模型各子模型:")
            for name, rmse in results['ensemble'].items():
                print(f"  - {name}: {rmse:.4f}")

    return results

if __name__ == "__main__":
    results = main()
    if results:
        print("\n🎉 训练测试完成!")
    else:
        print("\n⚠️ 训练测试未完成，请检查错误")
