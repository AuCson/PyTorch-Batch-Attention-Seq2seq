## A fast, batched Bi-RNN(GRU) encoder & attention decoder PyTorch implementation
Usage: Please refer to offical pytorch tutorial on attention-RNN machine translation, except that this implementation
handles batched inputs, and that it implements a slight different attention mechanism.<br>
To find out the formula-level difference of implementation, illustrations below will help a lot.<br>
<br>
PyTorch version mechanism illustration, see here: <br>
http://pytorch.org/tutorials/_images/decoder-network.png<br>
PyTorch offical Seq2seq machine translation tutorial:<br>
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html<br>
Bahdanau attention illustration, see here:<br>
http://images2015.cnblogs.com/blog/670089/201610/670089-20161012111504671-910168246.png<br>
<br>
PyTorch version attention decoder fed "word_embedding" to compute attention weights,
while in the origin paper it is supposed to be "encoder_outputs". In this repository, 
we implemented the origin attention decoder according to the paper(The only difference is that activation:tanh is omitted)<br>
<br>
## Speed up with batched tensor manipulation
PyTorch supports element-wise fetching and assigning tensor values during procedure, but actually it is slow especially when running on GPU. In a tutorial(https://github.com/spro/practical-pytorch),
attention values are assigned element-wise; it's absolutely correct(and intuitive from formulas in paper), but slow on our GPU.
Thus, we re-implemented a real batched tensor manipulating version, and it achieves <b>more than 10X speed improvement.</b><br>
<br>
This code works well on personal projects.

