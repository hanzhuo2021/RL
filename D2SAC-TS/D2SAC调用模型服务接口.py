# 强化学习模型接口文件，服务于平台的调用

import logging

import numpy as np
import paddle
from flask import Flask, request, jsonify
from paddle.distribution import Normal

from Actor import Actor


def get_action(state):
    # 初始化模型
    actor = Actor(48, 1)
    # 载入模型参数
    actor_state_dict = paddle.load("D2SAC.pdparams")
    # 将load后的参数与模型关联起来
    actor.set_state_dict(actor_state_dict)

    # 开始模型求解
    state = np.array(state)
    state = paddle.to_tensor(state.reshape(1, -1), dtype='float32')
    act_mean, act_log_std = actor(state)
    normal = Normal(act_mean, act_log_std.exp())
    # 重参数化  (mean + std*N(0,1))
    x_t = normal.sample([1])
    action = paddle.tanh(x_t)

    log_prob = normal.log_prob(x_t)
    log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
    log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)

    action, _ = action[0], log_prob[0]

    action_numpy = action.cpu().numpy()[0]

    action = action_numpy

    # 动作转换
    if action <= - 1.0 / 2:
        action = 0
    elif action <= 0.0:
        action = 1
    elif action <= 1.0 / 2:
        action = 2
    else:
        action = 3

    return action


app = Flask(__name__)


@app.route('/getD2SACAction', methods=['POST'])
def getD2SACAction():
    try:
        state = request.get_json()
        responses = get_action(state)
    except IndexError as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    except KeyError as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    except ValueError as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    except Exception as e:
        logging.error(str(e))
        return 'exception:' + str(e)
    else:
        return jsonify(responses)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
