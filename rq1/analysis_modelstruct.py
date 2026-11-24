import sys
import os
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import load_model

def analyze_model(model):
    layer_counts = defaultdict(int)
    
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_counts[layer_type] += 1

    return layer_counts

def save_results(model_name, layer_counts):
    os.makedirs("results", exist_ok=True)
    result_path = os.path.join("results", f"{model_name}.txt")
    with open(result_path, "w") as f:
        f.write(f"模型层统计：{model_name}\n")
        for layer_name, count in layer_counts.items():
            f.write(f"{layer_name}: {count}\n")
    print(f"统计结果已保存到 {result_path}")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python 文件名.py model_name.h5")
        return

    model_name = sys.argv[1]
    model_path = os.path.join('..', 'model', f'{model_name}.h5')

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    layer_counts = analyze_model(model)
    save_results(model_name, layer_counts)

if __name__ == "__main__":
    main()
