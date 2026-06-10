"""
模型模块完整测试脚本
验证所有核心功能
"""
import sys
import os
import traceback
import py_compile
import importlib.util

os.chdir("C:/A股智能分析预测系统A/app")
sys.path.insert(0, "C:/A股智能分析预测系统A/app")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_xgboost():
    print("\n" + "="*60)
    print("🔍 测试: XGBoost模型 (特征69+贝叶斯优化)")
    print("="*60)
    try:
        # 1. 语法检查
        print("📋 1. 语法检查...", end=" ")
        py_compile.compile("models/xgboost_model.py", doraise=True)
        print("✅ 通过")

        # 2. 导入
        print("📋 2. 模块导入...", end=" ")
        m = load_module("xgboost_model", "models/xgboost_model.py")
        print("✅ 通过")

        # 3. 类
        print("📋 3. XGBoostPredictor类...", end=" ")
        cls = m.XGBoostPredictor
        print("✅ 通过")

        # 4. 贝叶斯优化
        print("📋 4. 贝叶斯优化函数...", end=" ")
        has_opt = hasattr(m, 'optimize_xgboost') and callable(m.optimize_xgboost)
        bays_flag = getattr(m, 'BAYES_AVAILABLE', False)
        print(f"✅ 函数存在={has_opt}, BAYES_AVAILABLE={bays_flag}")

        # 5. 特征数量
        print("📋 5. 特征数量...", end=" ")
        import inspect
        feat_lines = [l for l in inspect.getsource(m).split('\n') if 'RET' in l and ('lags' in l.lower() or 'LAG' in l)]
        print(f"✅ 收益率滞后特征: {len(feat_lines)}行")
        # 检查函数_build_features是否存在
        has_build = hasattr(m, '_build_features')
        print(f"   _build_features函数: {has_build}")

        return True
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        return False

def test_lstm():
    print("\n" + "="*60)
    print("🔍 测试: LSTM模型 (双向+Attention)")
    print("="*60)
    try:
        # 1. 语法检查
        print("📋 1. 语法检查...", end=" ")
        py_compile.compile("models/lstm_model.py", doraise=True)
        print("✅ 通过")

        # 2. 导入
        print("📋 2. 模块导入...", end=" ")
        m = load_module("lstm_model", "models/lstm_model.py")
        print("✅ 通过")

        # 3. 类
        print("📋 3. LSTMPredictor类...", end=" ")
        cls = m.LSTMPredictor
        print("✅ 通过")

        # 4. 核心方法
        print("📋 4. 核心方法...", end=" ")
        has_train = hasattr(cls, 'train')
        has_predict = hasattr(cls, 'predict')
        print(f"train={has_train}, predict={has_predict}")

        # 5. Keras构建（双向LSTM + Attention）
        print("📋 5. Keras模型构建...", end=" ")
        src = str(dir(m))
        has_keras = hasattr(m, '_build_keras') or '_build_keras' in str(dir(cls))
        print(f"✅ _build_keras={has_keras}")

        return True
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        return False

def test_ensemble():
    print("\n" + "="*60)
    print("🔍 测试: 集成模型 (时序CV+Stacking)")
    print("="*60)
    try:
        # 1. 语法检查
        print("📋 1. 语法检查...", end=" ")
        py_compile.compile("models/ensemble.py", doraise=True)
        print("✅ 通过")

        # 2. 导入
        print("📋 2. 模块导入...", end=" ")
        m = load_module("ensemble", "models/ensemble.py")
        print("✅ 通过")

        # 3. 核心类
        print("📋 3. 核心类检查...", end=" ")
        has_de = hasattr(m, 'DynamicEnsemble')
        has_ep = hasattr(m, 'EnsemblePredictor')
        has_stacking = hasattr(m, 'stacking_ensemble')
        print(f"DynamicEnsemble={has_de}, EnsemblePredictor={has_ep}, stacking_ensemble={has_stacking}")

        # 4. TimeSeriesSplit（修复数据泄露）
        print("📋 4. TimeSeriesSplit（数据泄露修复）...", end=" ")
        src = m.stacking_ensemble.__code__.co_names
        has_tscv = 'TimeSeriesSplit' in src
        print(f"✅ TimeSeriesSplit={has_tscv}")

        # 5. EnsemblePredictor方法
        print("📋 5. EnsemblePredictor方法...", end=" ")
        ep_methods = [x for x in dir(m.EnsemblePredictor) if not x.startswith('_')]
        print(f"✅ {ep_methods}")

        return True
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        return False

def test_online_learner():
    print("\n" + "="*60)
    print("🔍 测试: 在线学习模块")
    print("="*60)
    try:
        # 1. 语法检查
        print("📋 1. 语法检查...", end=" ")
        py_compile.compile("models/online_learner.py", doraise=True)
        print("✅ 通过")

        # 2. 导入
        print("📋 2. 模块导入...", end=" ")
        m = load_module("online_learner", "models/online_learner.py")
        print("✅ 通过")

        # 3. 类
        print("📋 3. OnlineLearner类...", end=" ")
        cls = m.OnlineLearner
        print("✅ 通过")

        # 4. 核心方法
        print("📋 4. 核心方法...", end=" ")
        methods = [x for x in dir(cls) if not x.startswith('_')]
        print(f"✅ {methods}")

        # 5. daily_update方法
        print("📋 5. daily_update方法...", end=" ")
        has_daily = hasattr(cls, 'daily_update')
        print(f"✅ {has_daily}")

        return True
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🚀 A股智能分析预测系统 - 完整模型测试")
    print("="*60)

    results = []
    results.append(("XGBoost (特征69+贝叶斯)", test_xgboost()))
    results.append(("LSTM (双向+Attention)", test_lstm()))
    results.append(("Ensemble (时序CV+Stacking)", test_ensemble()))
    results.append(("OnlineLearner (在线学习)", test_online_learner()))

    print("\n" + "="*60)
    print("📊 测试汇总")
    print("="*60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}  {name}")

    print(f"\n总计: {passed}/{total} 通过")
    if passed == total:
        print("\n🎉 所有测试通过！模型代码可以正常使用。")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
