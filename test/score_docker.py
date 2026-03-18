"""
本脚本用于将预测出的五支股票与实际的股票数据进行对比，计算加权收益，形成最终得分。
"""
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Calculate stock prediction score.')
parser.add_argument('team_name', type=str, help='The name of the team (used for file naming).')
args = parser.parse_args()
output_path = f'./test/results_output/{args.team_name}.csv'
test_data_path = './data/test.csv'
def is_valid_prediction(test_data):
    """
    验证选手输出的结果是否合法：需要包含最多五支股票，并且权重之和为1.
    """
    is_valid = True
    if len(test_data) > 5:
        is_valid = False
    weight_sum = test_data['weight'].sum()
    if weight_sum != 1:
        is_valid = False
    if not is_valid:
        raise ValueError(f"预测结果不合法：最多只能包含五支股票，并且权重之和必须为1. 当前权重之和为 {weight_sum}.")
def calculate_return(group):
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']
def calculate_predict_weight_score(output_data, test_data):
    # 选择输出指定的5个股票
    test_data = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
    # 只选最后五个记录
    test_data = test_data.groupby('股票代码').tail(5)
    # 分别计算收益率
    group = test_data.groupby('股票代码')
    result = group.apply(calculate_return).reset_index().rename(columns={0: '收益率'})
    result = result.merge(output_data, on='股票代码')
    # 计算加权收益率
    final_score = (result['收益率'] * result['权重']).sum()
    return final_score
def calculate_optimal_score(test_data):
    # 计算每只股票的收益率
    group = test_data.groupby('股票代码').tail(5).groupby('股票代码')
    result = group.apply(calculate_return).reset_index().rename(columns={0: '收益率'})
    # 选择收益率最高的5只股票
    top_stocks = result.nlargest(5, '收益率')
    # 计算最优加权收益率
    optimal_score = (top_stocks['收益率'] * 0.2).sum()
    return optimal_score
def calculate_average_score(test_data):
    # 计算每只股票的收益率
    group = test_data.groupby('股票代码').tail(5).groupby('股票代码')
    result = group.apply(calculate_return).reset_index().rename(columns={0: '收益率'})
    # 计算平均收益率
    average_score = result['收益率'].mean()
    return average_score
# 读取测试数据
try:
    test_data = pd.read_csv(test_data_path)
    is_valid_prediction(test_data)
except Exception as e:
    print(f"Error reading test data or validating prediction: {e}")
    # 保存结果到 CSV 文件
    result = pd.DataFrame(
        {
            "Team Name": [args.team_name],
            "Final Score": [-999],
        }
    )
    result.to_csv("./temp/tmp.csv", index=False)



test_data = test_data[['股票代码', '日期', '开盘', '收盘']]
# 读取输出数据
output_data = pd.read_csv(output_path).rename(columns={'stock_id': '股票代码', 'weight': '权重'})

# 第一步：计算预测股票的加权收益率
predict_weight_score = calculate_predict_weight_score(output_data, test_data)


# 保存结果到 CSV 文件
result = pd.DataFrame(
    {
        "Team Name": [args.team_name],
        "Final Score": [predict_weight_score],
    }
)
result.to_csv("./temp/tmp.csv", index=False)