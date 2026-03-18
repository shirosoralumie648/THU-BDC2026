# 配置环境
提前安装uv
可以在有conda的情况下直接pip install uv或者参考互联网上uv的安装教程
1) 使用 `uv` 安装依赖
`uv sync`

2) 激活虚拟环境
linux
`source .venv/bin/activate`
windows
`\.venv\Script\activate`

# 准备数据
在get_stock_data.py中修改：
start_date = "***"
end_date = "***"
运行`python get_stock_data.py`即可下载所需时间段的数据（如果出现网络问题，请关闭代理重试）

在测试最终的得分时，需要先将数据划分为训练数据和测试数据，一般将数据的最后5个交易日设置为test数据，之前的数据为训练数据

修改data/split_train_test.py中的parse_args函数中设置训练集开始，结束时间，测试集的开始，结束时间。

运行`python data/split_train_test.py`，即可在data目录下生成train.csv与test.csv

# 训练与测试

训练：运行根目录下的sh train.sh

测试：运行根目录下的sh test.sh

训练和测试都完成之后，会在output目录下生成result.csv，即为预测的后五个交易日收益率最高的五个股票及权重

得到result.csv之后，可以运行`python test/score_self.py`，将选手的预测股票与测试集比较，在计算出最终的加权收益率，作为选手可自行参考的得分，默认保存在/temp/tmp.csv。

**注意**目前代码中直接选择排序得分最高的五个股票，平均权重。选手可以自行更改代码逻辑，只要满足最多五个股票，权重之和为1即可。

# 打包docker
在训练与测试完成之后，需要首先将项目整体打包成一个docker镜像（打包），再将该镜像导出为一个.tar文件（导出），最终提交该tar文件即可，里面需要包含运行时的所有环境及依赖，具体可以参考或修改Dockerfile

## docker镜像创建
镜像创建指令：

如果出现网络请求错误，请尝试使用代理或关闭代理。

`docker buildx build  --platform linux/amd64  --build-arg IMAGE_NAME=nvidia/cuda  -t bdc2026 .`

## 镜像导出
`docker save -o 队伍名称.tar bdc2026:latest`

# 可运行性验证
选手根据需要可以进行三步运行验证，包括：
1. 本机环境直接运行验证

需要选手保证根目录的train.sh, test.sh成功运行，并且在output目录下生成一个result.csv

2. 对打包后的docker进行完整运行验证

在这一步，选手需要按照打包docker的流程先将项目打包为一个镜像（对应上面的docker镜像创建），暂时不用导出会.tar文件

然后在根目录下运行`docker compose up`。这一下会对打包成的docker是否可运行进行验证，如果运行后在test/output中得到result.csv，验证即成功。

3. (可选)模拟赛事方最终以批处理的方式进行打分验证

这一步需要选手将镜像导出为tar文件（最终需要选手提交的文件），然后将tar文件放到test/tars目录下，并将.tar文件名写入test/tar_files_list.txt文件中。（比如tar文件叫1.tar，则在tar_files_list.txt的第一行写入1.tar即可）

然后在根目录下运行
linux
`python test/test.py`
windows
`python test/test_windows.py`

成功运行后，如果在test/result.csv中看到，类似下面的结果，该步验证成功。
```
Team Name,Final Score
1,-0.0397556344883805
```