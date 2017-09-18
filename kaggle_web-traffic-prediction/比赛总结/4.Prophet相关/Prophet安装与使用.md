### 安装
根据 github 源项目介绍， prophet 依赖于 pystan，故先安装 PyStan
#### PyStan
首先在 linux 下安装 anaconda
- 下载 Anaconda2-4.4.0-Linux-x86_64.sh
- 运行 ``` zsh ~/所在目录/Anaconda2-4.4.0-Linux-x86_64.sh``` (我的terminal用的是 zsh，不是bash)
- 一路 yes，中间有一步是询问是否要把 anaconda 路径加入 PATH，选择 yes，否则需要自己去 ~/.zshrc 中加入路径 
```
# added by Anaconda2 4.4.0 installer 
export PATH="/home/用户/anaconda2/bin:$PATH"
```
- 最后，``` source ~/.zshrc``` 使生效
- 用```conda list```测试是否成功

安装 PyStan 的依赖(anaconda安装后这些已经安装)
- numpy
- cython

安装 PyStan
- 输入```pip install pystan```
If you are using a VM, be aware that you will need at least 2GB of memory to run PyStan.

#### Prophet
成功安装 pystan 后，输入 ```pip install fbprophet```
失败的话，用这个```conda install -c conda-forge fbprophet```

### 使用
Prophet 遵循 sklearn API，建立一个 Prophet 实例，然后用 fit 和 predict 方法
#### Quick Start
**input**: ds 和 y. ds (datestamp) 必须包含一个 date / datetime. y 是一个数值，代表着需要预测的量
这里用的是 prophet 自带的例子对 Peyton Manning 维基百科页面的预测，它呈现了一些prophet 模型的特性
- multiple seasonality
- changing growth rates
- the ability to model special days
<img src="http://7xub54.com1.z0.glb.clouddn.com/img/github/ds_y.png">

**创建 Prophet 对象**
```
m = Prophet()
m.fit(df)
```
**构造需要预测的 dataframe**
```
future = m.make_future_dataframe(periods=365)
```
**预测 + output**
```
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```
这个 dataframe 包含 预测出的值 yhat 、它的不确定范围以及一些 components,比如 trend_upper/weekly/seasonal...
<img src="http://7xub54.com1.z0.glb.clouddn.com/img/github/pro_result.png">

**作图**
```
m.plot(forecast)
m.plot_components(forecast)
```
<img src="http://7xub54.com1.z0.glb.clouddn.com/img/github/res1.png">
<img src="http://7xub54.com1.z0.glb.clouddn.com/res2.png">
 
### Reference
1. https://github.com/facebookincubator/prophet  github主页
2. http://pystan.readthedocs.io/en/latest/installation_beginner.html 
3. https://docs.continuum.io/anaconda/install/linux 
4. https://facebookincubator.github.io/prophet/docs/ 使用