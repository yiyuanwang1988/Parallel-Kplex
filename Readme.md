# ParaKplex: A Parallel Local Search Algorithm for the Maximum K-Plex Problem

We propose a parallel local search algorithm, ParaKplex, for solving the maximum k-plex problem.


## 1. compile

To compile the source code and generate the executable ./ParaKplex, use the following command:
```
g++ -O3 -pthread -o ParaKplex splex_big.cpp
```



## 2. Usage

To execute the code, use the following command format:
```
./ParaKplex -f <filename> -s <parameter> -t <max seconds> [-o optimum object] [-n threads]
```
**Note:** ```<>``` indicates required parameters; ```[]``` indicates optional parameters. The default value for ```-o``` is ```999999```, and the default for ```-n``` is ```10``` threads.

### **For example**

```
./ParaKplex -f ./DIMACS10/fe-4elt2 -k 3 -t 1000
```

**Output**

```
instance ./DIMACS10_84/fe-4elt2 k 3 size 6 time 0.000167129
solution: 5808 7847 7846 7946 7845 7947
```
