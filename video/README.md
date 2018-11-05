## C3D 特征提取

1.  Github Page：https://github.com/facebook/C3D
2. Project Page: http://vlg.cs.dartmouth.edu/c3d/
3. User Guide: https://docs.google.com/document/d/1-QqZ3JHd76JfimY4QKqOojcEaf5g3JS0lNh-FHTxLag/edit

自己要提取的话，先仔细看下 User Guide，他提供了两种输入数据的方式。

一种是先提取视频帧，然后自己要预先生成两个文件。一个是后缀名称为 **.list** 的文件，用于提取是作为输入，里面的格式如下：
![](https://ws1.sinaimg.cn/large/006tNbRwgy1fwwz744i6hj316o0gyap9.jpg)。

另一个文件则是后缀名为 **.prefix** 用于对输出特征文件的名字命名。


### Caffe
编译的 Caffe 的时候，protobuf 版本我这里是 2.5 版本，高版本如最新的 3.6 版本可能会出错。