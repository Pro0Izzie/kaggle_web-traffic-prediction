import pandas as pd
import os

# 将预测生成的多个csv连接为一个文件


sub_df = pd.DataFrame(columns=['Id', 'Visits'])
rootDir = u'子csv文件所在文件夹路径'
dir_list = os.listdir(rootDir)
dir_id = [int(filename.replace('.csv','')) for filename in dir_list]
dir_dict = dict(zip(dir_id, dir_list))
dir_tuple = sorted(dir_dict.iteritems(), key=lambda x:x[0])
for dir_num, lists in dir_tuple: 
    path = os.path.join(rootDir, lists) 
    single_df = pd.read_csv(path)
    sub_df = pd.concat([sub_df,single_df],ignore_index=True)
# print sub_df.shape
sub_df.to_csv(u'合成csv的路径', index=False)