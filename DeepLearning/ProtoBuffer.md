# ProtoBuffer

2021-12-29
google内部使用的一种数据编码格式，二进制存储，之前使用过，但是具体跟json相比优劣在哪没有系统整理过，这里梳理一下

## 效率

Protocol buffers或Protobuf是由Google开发的二进制格式，用于在不同服务之间序列化数据。 Google开源了这个协议，支持JavaScript，Java，C＃，Ruby等。 这篇文章的测试数据显示Protocol buffers的传输效率比JSON快6倍具体跟json相比优劣在哪没有系统整理过，这里梳理一下


https://o-u-u.com/?p=1765 这篇文章讲的比较清楚，对比了json，java serilizable 和 protobuf

拿了一个很简单的例子，一个user类，里面有string变量name，double变量age，和list变量friends。
1. 首先在java里建了一个类，然后构建了一个user对象，传入上面的变量值，使用google的gson包将java对象转为json，可以
看到json结构非常清晰，字节数153
2. 同样用上面的java类，implement Serializable接口，带序列化uid，然后测试序列化和反序列化，字节数238。
这个优点是不需要额外的包，java原生支持，但是无法跨语言，字节数较大，对象属性变化比较敏感
3. 先使用google的protobuf包，建立一个proto类，在里面定义好结构，然后用protobuf包编译这个proto文件得到一个java类builder。
然后在编解码的时候，就使用这个builder，传值，build，解码的时候同样用这个生成的java类parse，字节数53
很明显，protobuf编码字节短，支持跨语言。不过需要额外的工具生成代码。

现在回想起来了之前做的数据采集的sdk，就是这样用的，其实很简单。就是写proto，然后编译成java类，用来把数据填进去做编解码。
压缩的结果都是二进制的。
