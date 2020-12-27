<font face="Times New Roman">

# Path planning of Cleaning vehicle

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#Contributing)

## Background

The program is based on the competition in convex optimization (Fall 2020) class, which requires a shortest path for a cleaning vehicle covering all rubbishes and avoiding all polygons given. In addition, sampling waypoints should be less than 100 and the turning angle of the vehicle should not be larger than ![](http://latex.codecogs.com/svg.latex?\\0.2\pi).

## Install

This module depends upon pytorch and OpenCV, which should be installed before running the program.

```
pip install torch
pip install opencv-python
```



## Usage

Run *solver.py* directly to use the program.

```
python solver.py
```

### Define variables and Functions

First define the visualization of the problem itself and the path calculated can be shown in the figure.

```python
def visualize(path, ind=None):
```

Then define the calculation of the angle of two lines and how to interpolate sampling points to satisfy the constraint of turning angle.

```python
def theta(line1, line2):
def theta_adjust(path):
```

To optimize the objective function,  class variable Solver is defined with init, collision and lpath part.

```python
class Solver(nn.Module):
    def __init__(self, starting, ending, point_num):
        super(Solver, self).__init__()
    def rectangle(self, p):
    def collision_fxy(self):
    def lpath(self):    
```

### Optimization process

To optimize the shortest path, first use dynamic programming to get the sequence of rubbishes to be cleaned, which is included in the directory *python_tsp*. Then sort the rubbish points for further optimization.

```python
from python_tsp.exact import solve_tsp_dynamic_programming
permutation, distance = solve_tsp_dynamic_programming(dis)
points = rubbish[np.array(permutation)]
```

For each two rubbish points, a shortest path without collision with polygons is obtained by Adam algorithm and the connecting of these paths is a optimized path.

```python
opt2 =  torch.optim.Adam(solver.parameters(), lr=0.5)
y = solver.lpath() + 1e3 * solver.collision_fxy()
opt2.zero_grad()
y.backward()
opt2.step()
```

Finally, adjust turning angles by *theta_adjust* to satisfy all the constraints.

```python
all_path = theta_adjust(path)
```

### Result Display

Use *visualize* function to show the result in the picture. Calculate the final path length.

```python
print(all_path.shape)
im = visualize(all_path)
all_dis = np.sum(all_path**2, 1)
all_dis = np.sqrt(all_dis)
all_dis = np.sum(all_dis)
print(all_dis)
```

The final result is shown in the following figure.

![image](https://github.com/zhenyuw16/path-planning/blob/main/result.png)

## Contributing

The contributors of this program is shown in [Contributors](https://github.com/zhenyuw16/path-planning/graphs/contributors). 



