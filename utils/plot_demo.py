import pandas as pd
from tqdm import trange


# 假设你有一个函数来计算和更新这些指标
def update_metrics(step, loss, miou):
    # 如果 DataFrame 已经存在，则加载现有数据
    try:
        metrics_df = pd.read_excel('training_metrics.xlsx', index_col=None, header=0, engine='openpyxl')
    except FileNotFoundError:
        # 如果文件不存在，则创建一个新的 DataFrame
        metrics_df = pd.DataFrame(columns=['Step', 'Loss', 'MIoU'])

    # 添加新的一行数据
    new_row = pd.DataFrame({'Step': [step], 'Loss': [loss], 'MIoU': [miou]})
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    # 将 DataFrame 写入 Excel 文件
    with pd.ExcelWriter('training_metrics.xlsx', mode='a', if_sheet_exists='overlay') as writer:
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')


if __name__ == '__main__':
    data = {
        'epoch': [1, 2, 3],
        'step': [1],
        'train_loss': [1, 2],
        'dice_score': [],
        'miou': [],
    }
    # 填充缺失值
    data['dice_score'] = [None] * len(data['epoch'])  # 使用None填充
    data['miou'] = [None] * len(data['epoch'])  # 使用None填充
    data['train_loss'] = [None] * len(data['epoch'])  # 使用None填充
    data['step'] = [None] * len(data['epoch'])  # 使用None填充
    df = pd.DataFrame(data)
    df['dice_score'].fillna(value=0, inplace=True)  # 将NaN替换为0或其他默认值
    df['miou'].fillna(value=0, inplace=True)  # 将NaN替换为0或其他默认值
    df.to_excel(f'training_results_total_{123}.xlsx', index=False)
    print(df)
