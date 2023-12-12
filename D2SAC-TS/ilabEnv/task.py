# 初始化 taskCost : cpuCost、gpuCost、memoryCost、networkCost、execTime
task_cost_map = {}


def init_task_cost_map():
    node1_task_cost_map = {
        1: [5.265717959, 0.0, 0.0, 2.173451946, 0.0, 30.084],
        2: [9.412870204, 0.0, 0.0, 12.55689167, 0.0, 1296.356],
        3: [3.880934538, 0.0, 0.0, 1.865609422, 0.0, 37.352],
        4: [3.68405466, 0.0, 0.0, 2.76759557, 0.0, 10.35300000],
        5: [10.37894897, 0.0, 0.0, 0.50028653, 0.0, 25.0946],
        6: [6.50485634, 0.0, 0.0, 0.07832146, 0.0, 253.259],
        7: [12.5849409, 0.0, 0.0, 3.48616331, 0.0, 204.89300000],
        8: [11.3660398, 0.0, 0.0, 3.74294104, 0.0, 76.78550000],
        9: [4.07696857, 0.0, 0.0, 2.30817161, 0.0, 15.61500000],
        10: [6.14590384, 0.0, 0.0, 0.68468044, 0.0, 32.164],
        11: [1.3660398, 0.0, 0.0, 0.23423432, 0.0, 10.664],
        12: [6.42644771, 0.0, 0.0, 0.62714644, 0.0, 16.793],
        13: [14.5033179, 0.0, 0.0, 5.45867169, 0.0, 306.90100000],
        14: [3.78073885, 0.0, 0.0, 3.16038647, 1.533454228, 125.64300000],
        15: [6.9160563, 0.0, 0.0, 4.51657713, 2.85433927, 53.9729],
        16: [11.68162131, 0.0, 0.0, 6.31849177, 0.0, 822.367],
        17: [14.7310486, 0.0, 0.0, 4.0466973, 0.0, 957.75000000],
        18: [1.24732253, 0.0, 0.0, 9.18657313, 0.0, 21.4296],
        19: [2.049973879, 0.0, 0.0, 11.58516044, 0.0, 1674.138],
        20: [13.6063039, 0.0, 0.0, 8.12035315, 0.0, 1008.30800000],
        21: [6.83734259, 0.0, 0.0, 8.76380651, 1.683091102, 41.7691],
        22: [4.31430959, 0.0, 0.0, 2.467046, 0.0, 80.63250000],
        23: [8.55390507, 0.0, 0.0, 2.90822359, 0.0, 43.6857],
        24: [8.81176098, 0.0, 0.0, 1.11091539, 0.0, 37.012],
        25: [9.30762748, 0.0, 0.0, 3.13748926, 0.0, 101.57400000],
        26: [4.59612185, 0.0, 0.0, 1.48861603, 0.0, 180.935],
        27: [14.2652047, 0.0, 0.0, 4.67901485, 0.0, 920.31800000],
        28: [9.30762748, 0.0, 0.0, 3.13748926, 0.0, 101.57400000],
        29: [4.59612185, 0.0, 0.0, 1.48861603, 0.0, 180.935],
        30: [14.2652047, 0.0, 0.0, 4.67901485, 0.0, 920.31800000]
    }

    node2_task_cost_map = {
        1: [4.635114925, 0.0, 0.0, 2.659224477, 0.0, 24.192],
        2: [6.145970029, 0.0, 0.0, 14.15521474, 0.0, 1322.447],
        3: [9.16885166, 0.0, 0.0, 3.325948954, 0.0, 37.303],
        4: [2.53264207, 0.0, 0.0, 3.95403302, 0.0, 11.45350000],
        5: [10.6728618, 0.0, 0.0, 0.45367773, 0.0, 25.0348000],
        6: [10.13569333, 0.0, 0.0, 0.6197538, 0.0, 243.176],
        7: [10.6026423, 0.0, 0.0, 5.30526235, 0.0, 196.62250000],
        8: [9.99184266, 0.0, 0.0, 4.92089267, 0.0, 71.44800000],
        9: [2.7218035, 0.0, 0.0, 4.68862553, 0.0, 16.08300000],
        10: [3.03846337, 0.0, 0.0, 0.47790603, 0.0, 29.738],
        11: [0.61945655, 0.0, 0.0, 0.17841451, 0.0, 9.657],
        12: [3.0544512, 0.0, 0.0, 0.05277231, 0.0, 16.830],
        13: [12.9113279, 0.0, 0.0, 6.01694695, 0.0, 300.39800000],
        14: [5.32109291, 0.0, 0.0, 4.17958983, 1.687796969, 128.98666667],
        15: [11.11445986, 0.0, 0.0, 4.44889296, 2.841956469, 49.6189],
        16: [11.64695476, 0.0, 0.0, 5.834317078, 0.0, 825.456],
        17: [12.9164419, 0.0, 0.0, 5.17757351, 0.0, 958.00800000],
        18: [3.32516258, 0.0, 0.0, 8.45657345, 0.0, 20.3532],
        19: [2.154880627, 0.0, 0.0, 21.89700204, 0.0, 1666.208],
        20: [11.9654676, 0.0, 0.0, 9.91460704, 0.0, 1008.21700000],
        21: [3.96989573, 0.0, 0.0, 8.17847465, 1.676382816, 41.3386],
        22: [3.03201497, 0.0, 0.0, 4.52519984, 0.0, 80.08750000],
        23: [6.06755341, 0.0, 0.0, 3.85490562, 0.0, 38.2302],
        24: [6.11155733, 0.0, 0.0, 0.58376417, 0.0, 36.604],
        25: [7.2317911, 0.0, 0.0, 5.53086958, 0.0, 96.57366667],
        26: [8.79773871, 0.0, 0.0, 3.74294104, 0.0, 164.276],
        27: [13.6878027, 0.0, 0.0, 7.12482381, 0.0, 867.56800000],
        28: [13.6878027, 0.0, 0.0, 7.12482381, 0.0, 867.56800000],
        29: [13.6878027, 0.0, 0.0, 7.12482381, 0.0, 867.56800000],
        30: [13.6878027, 0.0, 0.0, 7.12482381, 0.0, 867.56800000]
    }

    node3_task_cost_map = {
        1: [4.736052068, 0.0, 0.0, 1.045374355, 0.0, 25.662],
        2: [7.745615862, 0.0, 0.0, 19.46172094, 0.0, 1395.674],
        3: [9.989909201, 0.0, 0.0, 3.18559079, 0.0, 38.805],
        4: [4.0010285, 0.0, 0.0, 2.69980363, 0.0, 12.68250000],
        5: [10.63129446, 0.0, 0.0, 0.68104104, 0.0, 25.9652],
        6: [5.23525437, 0.0, 0.0, 0.22977576, 0.0, 247.547],
        7: [13.3426084, 0.0, 0.0, 3.52677415, 0.0, 245.59900000],
        8: [11.6885389, 0.0, 0.0, 4.21407497, 0.0, 84.55450000],
        9: [6.1037322, 0.0, 0.0, 2.1825662, 0.0, 17.05700000],
        10: [4.32801918, 0.0, 0.0, 0.45908388, 0.0, 32.236],
        11: [2.0181407, 0.0, 0.0, 0.00447183, 0.0, 7.211],
        12: [4.75675012, 0.0, 0.0, 0.00379866, 0.0, 17.285],
        13: [14.627352, 0.0, 0.0, 5.35390004, 0.0, 348.51333333],
        14: [3.99894741, 0.0, 0.0, 3.19333314, 1.373078778, 135.52900000],
        15: [7.0696601, 0.0, 0.0, 2.95727819, 2.981054062, 58.0739],
        16: [10.65288569, 0.0, 0.0, 3.439422326, 0.0, 819.742],
        17: [15.0378188, 0.0, 0.0, 4.11037902, 0.0, 957.71650000],
        18: [1.57369116, 0.0, 0.0, 8.22862255, 0.0, 21.2201],
        19: [2.066398546, 0.0, 0.0, 10.98771849, 0.0, 1740.388],
        20: [14.0859286, 0.0, 0.0, 8.57192143, 0.0, 1008.53500000],
        21: [5.18987893, 0.0, 0.0, 4.60632178, 1.705993814, 45.5708],
        22: [4.51540557, 0.0, 0.0, 2.78511071, 0.0, 81.22800000],
        23: [7.42718608, 0.0, 0.0, 1.51329957, 0.0, 38.9709],
        24: [6.71075422, 0.0, 0.0, 1.39057947, 0.0, 38.227],
        25: [13.5340501, 0.0, 0.0, 3.4402163, 0.0, 114.43433333],
        26: [9.63760577, 0.0, 0.0, 3.74294104, 0.0, 147.479],
        27: [14.3913412, 0.0, 0.0, 5.77709092, 0.0, 959.73533333],
        28: [13.5340501, 0.0, 0.0, 3.4402163, 0.0, 114.43433333],
        29: [9.63760577, 0.0, 0.0, 3.74294104, 0.0, 147.479],
        30: [14.3913412, 0.0, 0.0, 5.77709092, 0.0, 959.73533333]
    }

    node4_task_cost_map = {
        1: [1.347234321, 0.0, 0.0, 0.045374355, 0.0, 6.372],
        2: [3.347234321, 0.0, 0.0, 2.4413423, 0.0, 244.598],
        3: [0.723122132, 0.0, 0.0, 0.32323235, 0.0, 6.233],
        4: [0.0232432, 0.0, 0.0, 0.05653432, 0.0, 0.825],
        5: [5.5341343, 0.0, 0.0, 0.30324324, 0.0, 24.748],
        6: [3.65432453, 0.0, 0.0, 0.3645654, 0.0, 24.737],
        7: [7.3545234, 0.0, 0.0, 0.2044553, 0.0, 34.765],
        8: [1.347234321, 0.0, 0.0, 1.49234645, 0.0, 8.415],
        9: [0.0123545, 0.0, 0.0, 0.0855424, 0.0, 1.311],
        10: [0.09356456, 0.0, 0.0, 0.05343234, 0.0, 3.418],
        11: [0.0324724, 0.0, 0.0, 0.00232532, 0.0, 0.798],
        12: [0.01723721, 0.0, 0.0, 0.02023432, 0.0, 1.710],
        13: [6.633454, 6.532433, 27.0917345, 5.4634343, 0.0, 22.951],
        14: [2.14324334, 7.0866343, 34.047455, 0.045374355, 1.68234348, 27.284],
        15: [5.73455234, 8.1232538, 34.625242, 0.045374355, 2.76845462, 94.587],
        16: [5.9234364, 4.0102666, 36.284545, 4.9823434, 0.0, 35.799],
        17: [1.0343264, 17.4704, 17.9809, 1.00423438, 0.0, 8.446],
        18: [0.9324343, 12.345433, 16.4674554, 4.734544, 0.0, 5.588],
        19: [1.92343434, 32.00285, 48.3423470, 5.8343443, 0.0, 32.480],
        20: [2.7543454, 36.653455, 10.343434, 4.54543543, 0.0, 27.502],
        21: [2.53344343, 45.146454, 48.654523, 1.00645234, 1.79345434, 61.066],
        22: [2.62344343, 11.094656, 36.01310, 1.89454543, 0.0, 73.030],
        23: [2.54545445, 22.9501, 29.3094758, 3.89233645, 0.0, 16.426],
        24: [0.7844564, 6.0865345, 10.73543, 0.045374355, 0.0, 4.140],
        25: [1.74345434, 25.392, 5.1645642, 1.64566535, 0.0, 6.941],
        26: [1.25635654, 25.3848, 24.3195565, 4.6745645, 0.0, 22.121],
        27: [7.5634541, 23.456645, 17.6159343, 2.074565, 0.0, 131.322],
        28: [1.74345434, 25.392, 5.1645642, 1.64566535, 0.0, 6.941],
        29: [1.25635654, 25.3848, 24.3195565, 4.6745645, 0.0, 22.121],
        30: [7.5634541, 23.456645, 17.6159343, 2.074565, 0.0, 131.322]
    }

    task_cost_map['node1'] = node1_task_cost_map
    task_cost_map['node2'] = node2_task_cost_map
    task_cost_map['node3'] = node3_task_cost_map
    task_cost_map['node4'] = node4_task_cost_map

    return task_cost_map

def get_task_list():
    # TASK_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    #                   27, 28, 29, 30]
    TASK_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    return TASK_LIST

def get_NODE_LIST():
    NODE_LIST = ["node1", "node2", "node3", "node4"]
    return NODE_LIST

# 输出内容：二维数组，每行为在各个节点上消耗的资源
# [[4.0010285, 0.0, 2.69980363, 0.0, 12.68250000],
#  [4.0010285, 0.0, 2.69980363, 0.0, 12.68250000],
#  [4.0010285, 0.0, 2.69980363, 0.0, 12.68250000]]
# def get_task_info(task_id):
#     task_info = {}
#     for key in task_cost_map:
#         node_task_cost_map = task_cost_map[key]
#         task_cost = node_task_cost_map[task_id]
#         task_info.extend(task_cost)
#     return task_info