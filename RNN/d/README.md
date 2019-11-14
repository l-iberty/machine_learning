# Using a dynamic RNN
使用tensorflow的API重写**动态**RNN模型，这样写出来的代码很玄妙，不如“**静态**”易于理解。

之前，我们在执行tensorflow graph之前手动地为每个timestep添加计算节点。这称为**静态**构造。我们也可以让tensorflow在执行时动态地创建计算图，这将变得更有效率。为了实现这一目标，我们把几个关键变量都reshape成`[num_batches, epoch_size, features]`的3维形式，然后使用tensorflow的`dynamic_rnn`函数。