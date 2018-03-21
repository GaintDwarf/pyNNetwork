import pyNNetwork as net

XI = [[0, 0], [0, 1], [1, 0], [1, 1]]
XO = [[1, 0], [0, 1], [0, 1], [1, 0]]

nn = net.NNetwork([2, 4, 3, 2])

nn.feedforword([0,0]) # expect random
nn.feedforword([1,0]) # expect random
nn.feedforword([0,1]) # expect random
nn.feedforword([1,1]) # expect random

for i in range(10):
        nn.SGradient_descent(XI * 12, XO * 12, 4, 3, 1000)
        nn.SGradient_descent(XI * 12, XO * 12, 4, 3, 1, XI, XO)

nn.feedforword([0,0]) # expect [=>1,=>0]
nn.feedforword([1,0]) # expect [=>0,=>1]
nn.feedforword([0,1]) # expect [=>0,=>1]
nn.feedforword([1,1]) # expect [=>1,=>0]