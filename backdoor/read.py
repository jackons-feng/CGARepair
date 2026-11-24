"""
功能：将 JSON 格式的神经元定位结果（localization results）
     转换为 backdoor_moo_2.py 可读取的 TXT 格式。

用法：
    python read.py <model_name> <epsilon> <seed>
示例：
    python read.py mnist 0.05 42

说明：
    - JSON 文件通常存放在 ../rq1/result/loc_result/<model_name>/ 下
    - 输出 TXT 文件（players.txt 与 backdoor_vals.txt）
      将保存在 ./loc_result/<model_name>/ 下
"""

import json
import os


def get_layer_index_mapping(model):
    """
    获取模型层索引的映射表。

    功能：
        将 Keras 模型中实际的 layer 索引（model.layers 中的下标）
        转换为按层类型顺序编号的索引（conv_0, conv_1, dense_0, ...）。
        主要用于将 loc_shapley_test.py 输出的层索引
        映射到 backdoor_moo_2.py 所期望的顺序。

    参数：
        model: Keras 模型对象

    返回：
        layer_to_seq: dict
            映射表，键为 (layer_type, actual_layer_idx)，
            值为对应的顺序编号，如：
            {('conv', 5): 2, ('dense', 18): 0}
    """
    layer_to_seq = {}
    conv_count = 0
    dense_count = 0
    
    # 遍历模型所有层
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            # 跳过无参数的层（如 Dropout、Flatten 等）
            continue
        
        lname = type(layer).__name__
        
        # 卷积层按出现顺序编号
        if 'Conv' in lname:
            layer_to_seq[('conv', i)] = conv_count
            conv_count += 1
        # 全连接层按出现顺序编号
        elif 'Dense' in lname:
            layer_to_seq[('dense', i)] = dense_count
            dense_count += 1
    
    return layer_to_seq


def convert_player_name(player, layer_to_seq):
    """
    将玩家名（神经元标识符）从“实际层索引”格式转换为“顺序层索引”格式。

    例如：
        输入：'conv_5_10'（表示第5层卷积层的第10个神经元）
        输出：'conv_2_10'（表示第2个卷积层的第10个神经元）

    参数：
        player: str，形如 'conv_5_10' 或 'dense_18_200'
        layer_to_seq: dict，由 get_layer_index_mapping() 生成的映射表

    返回：
        转换后的玩家名（若找不到对应层则原样返回）
    """
    parts = player.split('_')
    if len(parts) != 3:
        # 不符合格式则直接返回
        return player
    
    layer_type = parts[0]       # 'conv' 或 'dense'
    actual_layer_idx = int(parts[1])
    neuron_idx = parts[2]
    
    key = (layer_type, actual_layer_idx)
    if key in layer_to_seq:
        sequential_idx = layer_to_seq[key]
        return f'{layer_type}_{sequential_idx}_{neuron_idx}'
    else:
        print(f"Warning: Layer {layer_type}_{actual_layer_idx} not found in mapping")
        return player


def convert_json_to_txt(json_file, output_dir, model_path=None):
    """
    将 JSON 文件转换为 TXT 格式。

    输入：
        json_file: str，输入的 JSON 文件路径
        output_dir: str，输出目录
        model_path: str，可选参数（保留用于兼容旧代码，不使用）

    输出：
        在 output_dir 下生成两个文件：
        - players.txt: 所有神经元名称，逗号分隔
        - backdoor_vals.txt: 对应的数值（评分），逗号分隔
    """
    # ---- 1. 加载 JSON ----
    print(f"Loading JSON file: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # ---- 2. 读取关键字段 ----
    players = data['players']   # 神经元名列表
    scores = data['scores']     # 对应的重要性或影响值
    
    print(f"Found {len(players)} players and {len(scores)} scores")
    
    # ---- 3. 一致性检查 ----
    if len(players) != len(scores):
        print(f"Warning: Number of players ({len(players)}) != number of scores ({len(scores)})")
        min_len = min(len(players), len(scores))
        players = players[:min_len]
        scores = scores[:min_len]
        print(f"Truncated to {min_len} entries")
    
    # ---- 4. 若 JSON 已经是顺序编号格式，则直接使用 ----
    print("\nUsing player names directly from JSON (already in correct sequential format)")
    
    # ---- 5. 打印前10个样例以供核查 ----
    print("\nExample players:")
    for i in range(min(10, len(players))):
        print(f"  {players[i]}: {scores[i]}")
    
    # ---- 6. 创建输出目录 ----
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- 7. 写入 players.txt ----
    #players_file = os.path.join(output_dir, 'players.txt')
    players_file = os.path.join(output_dir, 'ccbr_players.txt')
    print(f"\nWriting players to: {players_file}")
    with open(players_file, 'w') as f:
        f.write(','.join(players))
    
    # ---- 8. 写入 backdoor_vals.txt ----
    #adv_vals_file = os.path.join(output_dir, 'backdoor_vals.txt')
    adv_vals_file = os.path.join(output_dir, 'ccbr_backdoor_vals.txt')
    print(f"Writing adv values to: {adv_vals_file}")
    with open(adv_vals_file, 'w') as f:
        f.write(','.join(map(str, scores)))
    
    # ---- 9. 打印总结信息 ----
    print(f"\nConversion complete!")
    print(f"  Players: {len(players)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files created:")
    #print(f"    - players.txt")
    #print(f"    - backdoor_vals.txt")
    print(f"    - ccbr_players.txt")
    print(f"    - ccbr_backdoor_vals.txt")


if __name__ == '__main__':
    import sys
    
    # ---- 命令行参数检查 ----
    if len(sys.argv) != 4:
        print("Usage: python read.py <model_name> <epsilon> <seed>")
        print("Example: python read.py mnist 0.05 42")
        sys.exit(1)
    
    # 参数说明：
    #   model_name: 模型名（如 gtsrb_resnet, fashion）
    #   epsilon: 用于兼容旧版接口，当前未使用
    #   seed: 随机种子，用于确定输入 JSON 文件名
    model_name = sys.argv[1]
    seed = sys.argv[2]
    
    # ---- 构造输入输出路径 ----
    #json_file = os.path.join('..', 'rq1', 'result', 'loc_result', model_name, 
                            #f'cfrepair_backdoor_512_alpha0_steps30_seed{seed}_topNone.json')
    json_file = os.path.join('..', 'rq1', 'result', 'loc_result', model_name, 
                            f'ccbr_backdoor_512_seed{seed}_topNone.json')
    output_dir = os.path.join('loc_result', model_name)
    
    # ---- 检查文件是否存在 ----
    if not os.path.exists(json_file):
        print(f"Error: Input file not found: {json_file}")
        sys.exit(1)
    
    # ---- 执行转换流程 ----
    try:
        convert_json_to_txt(json_file, output_dir)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)