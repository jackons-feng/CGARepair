import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

BATCH_SIZE = 1024

#功能：识别并分类模型中的层类型（卷积、全连接、批归一化）。
# 此函数遍历模型的所有层，根据层的类型和权重情况分配索引到相应的列表（conv, dense, batch）中。
# 这有助于后续精确地定位和操作这些层的权重。
def get_layer_index(init_model):
    conv = []
    dense = []
    batch = []
    j = 0
    for i, layer in enumerate(init_model.layers):
        weights = layer.get_weights()
        lname = type(layer).__name__
        if weights:
            if 'Batch' in lname:
                batch.append(j)
                j += 4
            elif 'Conv' in lname:
                conv.append(j)
                j += 2
            elif 'Dense' in lname:
                dense.append(j)
                j += 2
    return conv, dense, batch


def get_layer_indices(model):
    all_layers = model.layers
    all_weights = model.get_weights()

    weight_index = 0
    conv, dense, batch = [], [], []

    for layer in all_layers:
        weights = layer.get_weights()
        num_weights = len(weights)

        lname = type(layer).__name__

        if 'Batch' in lname:
            batch.append(weight_index)
        elif 'Conv' in lname:
            conv.append(weight_index)
        elif 'Dense' in lname:
            dense.append(weight_index)

        weight_index += num_weights

    return conv, dense, batch



#根据配置标志（是否考虑卷积层c_flag、全连接层d_flag），从模型中提取所有或部分权重，并将它们组织成玩家列表。
#如果之前已保存了玩家信息，直接从文件读取；否则，根据配置生成并保存玩家列表。这为后续的权重贡献分析提供基础。
def get_players(init_model=None, experiment_dir=None, c_flag=False, d_flag=False, convs=None, denses=None):
    if tf.io.gfile.exists(os.path.join(experiment_dir, 'players.txt')):
        with open(os.path.join(experiment_dir, 'players.txt')) as f:
            players = f.read().split(',')
        players = np.array(players)
    else:
        players = []
        weights = init_model.get_weights()
        if c_flag:
            for index, conv in enumerate(convs):
                players.append(['conv_{}_{}'.format(index, i) for i in range(weights[conv].shape[-1])])
        if d_flag:
            for index, dense in enumerate(denses):
                players.append(['dense_{}_{}'.format(index, i) for i in range(weights[dense].shape[-1])])
        players = np.sort(np.concatenate(players))
        with open(os.path.join(experiment_dir, 'players.txt'), 'w') as f:
            f.write(','.join(players))
    return players


def remove_players(var_vals, players, b_flag, convs, denses):
    if isinstance(players, str):
        players = [players]
    for player in players:
        if player.split('_')[0] == "conv":
            weight_index = convs[int(player.split('_')[1])]
            var_vals[weight_index][..., int(player.split('_')[-1])] = 0.  # conv weight
            if b_flag:
                var_vals[weight_index + 4][int(player.split('_')[-1])] = 1.  # batch norm variance
                var_vals[weight_index + 3][int(player.split('_')[-1])] = 0.  # batch norm mean
        else:
            weight_index = denses[int(player.split('_')[1])]
            var_vals[weight_index][..., int(player.split('_')[-1])] = 0.  # dense neuron weight
    return var_vals


def select_data(X, Y, key, num_examples):
    if key > -1:
        candidate_x = X[Y == key]
        candidate_y = Y[Y == key]
    else:
        candidate_x = X
        candidate_y = Y
    random_indices = np.random.choice(len(candidate_x), min(num_examples, len(candidate_x)), replace=False)
    return candidate_x[random_indices], candidate_y[random_indices]
