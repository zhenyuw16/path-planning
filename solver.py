import numpy as np 
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as sco
import torch
import torch.nn as nn

from python_tsp.exact import solve_tsp_dynamic_programming


def visualize(path):
    obstacles = [np.array(  ((1.23,3.47),(1.75,4.00),(2.10,3.63),(1.58,2.30),(1.40,2.67)) ),  
                          np.array(  ((4.65,5.98),(4.00,6.48),(4.52,7.68),(5.06,7.73),(5.90,6.95)) ),  
                          np.array(  ((6.78,3.40),(7.78,3.76),(7.78,5.10)) ), 
                          np.array(  ((4.00,3.00),(4.35,3.35),(4.80,3.45),(4.37,2.75)))  ]
    rubbishes = np.array((  (0.40,3.00),(2.90,2.20),(1.10,6.50),(2.90,5.00),(3.00,8.30),
                            (5.20,4.80),(7.40,7.40),(5.30,1.20),(7.80,2.60),(6.00,6.00),
                            (9.00,4.80),(5.00,8.50),(7.00,1.50),(2.50,7.50)   ))
    
    car = np.array([0.57/2, 0.7/2])
    car_width = 0.57
    car_length = 0.7
    r = np.array([[car[0]-car_width/2, car[1]-car_length/2], 
                      [car[0]+car_width/2, car[1]-car_length/2], 
                      [car[0]+car_width/2, car[1]+car_length/2], 
                      [car[0]-car_width/2, car[1]+car_length/2], ])
    
    im = np.zeros((1000,1000), dtype = np.uint8)
    point_obstacles = []
    for i in range(len(obstacles)):
        p = obstacles[i]
        p[:,0] = p[:,0]
        p[:,1] = 10 - p[:,1]
        point_obstacles.append(p)
    rr = r.copy()
    rr[:,0] = r[:,0]
    rr[:,1] = 10 - r[:,1]
    for i in range(len(point_obstacles)):
        cv2.fillConvexPoly(im, (point_obstacles[i]*100).astype('int'), 255)

    for i in range(len(rubbishes)):
        cv2.circle(im, (int(rubbishes[i][0]*100), int(1000 - rubbishes[i][1]*100)), 5, 255, -1)
    
    pp = path
    pp[:,1] = 10 - pp[:,1]
    
    for i in range(len(pp)):
        cv2.circle(im, (int(pp[i][0]*100), int(pp[i][1]*100)), 3, 200, -1)

    cv2.polylines(im, np.array([(pp*100)]).astype('int'), False, 128, 1)
    return im


class Solver(nn.Module):
    def __init__(self, starting, ending):
        super(Solver, self).__init__()
        self.obstacles = [torch.tensor(  ((1.23,3.47),(1.75,4.00),(2.10,3.63),(1.58,2.30),(1.40,2.67)) ),  
                          torch.tensor(  ((4.65,5.98),(4.00,6.48),(4.52,7.68),(5.06,7.73),(5.90,6.95)) ),  
                          torch.tensor(  ((6.78,3.40),(7.78,3.76),(7.78,5.10)) ), 
                          torch.tensor(  ((4.00,3.00),(4.35,3.35),(4.80,3.45),(4.37,2.75)))  ]
        self.starting = starting
        self.ending = ending

        line = ending - starting
        self.point_num = 7
        path = np.zeros((self.point_num,2))
        for i in range(1,self.point_num+1):
            path[i-1,:] = starting + i/self.point_num * line
        
        self.starting = torch.tensor(self.starting)
        self.ending = torch.tensor(self.ending)
        self.path =  nn.Parameter(torch.tensor(path[0:-1],  requires_grad=True))
        

    
    def rectangle(self, p):
        r = torch.tensor([[p[0]-self.car_width/2, p[1]-self.car_length/2], 
                      [p[0]+self.car_width/2, p[1]-self.car_length/2], 
                      [p[0]+self.car_width/2, p[1]+self.car_length/2], 
                      [p[0]-self.car_width/2, p[1]+self.car_length/2], 
                    ])
        return r


    def collision_wzy(self):
        path = torch.cat([self.starting[np.newaxis,:], self.path, self.ending[np.newaxis,:]], 0)
        p = path[0:-1]
        r = path[1:] - path[0:-1]
        colllist = []
        
        for i in range(len(self.obstacles)):
            ob = self.obstacles[i]
            sob = ob.shape[0]
            q = ob[:,:]
            s = ob[list(range(1, sob)) + [0]] - q
            qp = (q[np.newaxis,:,:]-p[:,np.newaxis,:])
            t = (qp[:,:,0] * s[np.newaxis,:,1] - qp[:,:,1] * s[np.newaxis,:,0]) / ( torch.transpose((r[np.newaxis,:,0] * s[:,np.newaxis,1] - r[np.newaxis,:,1] * s[:,np.newaxis,0]), 1, 0) + 1e-8)
            u = (qp[:,:,0] * r[:,np.newaxis,1] - qp[:,:,1] * r[:,np.newaxis,0]) / ( torch.transpose((r[np.newaxis,:,0] * s[:,np.newaxis,1] - r[np.newaxis,:,1] * s[:,np.newaxis,0]), 1, 0) + 1e-8)
            collt = 1 - (t/torch.abs(t) * (t-1)/torch.abs(t-1) + 1) / 2
            collu = 1 - (u/torch.abs(u) * (u-1)/torch.abs(u-1) + 1)  / 2
            coll = collt * collu
            colllist.append(coll)
        
        colllist = torch.cat(colllist, 1)
        return torch.sum(colllist) 
    
    def collision_fxy(self):
        path = torch.cat([self.starting[np.newaxis,:], self.path, self.ending[np.newaxis,:]], 0)
        p = path[0:-1]
        r = path[1:] - path[0:-1]
        sample=path[1:]
        sample_num=30
        for i in range(sample_num-1):
            sample_temp = p + r * (i+1) / sample_num
            sample=torch.cat((sample, sample_temp),dim=0)

        coll = 0

        for i in range(len(self.obstacles)):
            # print(linem, lineb)
            ob = self.obstacles[i]
            sob = ob.shape[0]
            q = ob[:, :]
            s = ob[list(range(1, sob)) + [0]] - q
            di = torch.cat((s[:, 1][:, np.newaxis], -s[:, 0][:, np.newaxis]), 1)
            di_norm = torch.norm(di, dim=1, keepdim=True)
            di = di / di_norm

            sample_q=-(q[np.newaxis, :, :] - sample[:, np.newaxis, :])
            #t=sample_q*di[np.newaxis, :, :].repeat(sample_num*99,1,1)
            #print(sample_q.shape, di.shape)
            temp=torch.relu(torch.sum((sample_q*di[np.newaxis, :, :].repeat(sample_num*self.point_num,1,1)),dim=2))
            co=torch.prod(temp,dim=1)
            coll += torch.sum(co)
        return coll
    
    def lpath(self):
        path = torch.cat([self.starting[np.newaxis,:], self.path, self.ending[np.newaxis,:]], 0)
        xp = path[1:] - path[0:-1]
        ll = torch.sum(torch.sqrt(1e-6 + torch.sum(xp**2, 1)))
        return ll
    


rubbish = np.array((  (0.57/2,0.7/2), (0.40,3.00),(2.90,2.20),(1.10,6.50),(2.90,5.00),(3.00,8.30),
                           (5.20,4.80),(7.40,7.40),(5.30,1.20),(7.80,2.60),(6.00,6.00),
                           (9.00,4.80),(5.00,8.50),(7.00,1.50),(2.50,7.50)   ))


dis = rubbish[:,np.newaxis,:] - rubbish[np.newaxis,:,:]
dis = np.sqrt(np.sum(dis**2, 2))
permutation, distance = solve_tsp_dynamic_programming(dis)
points = rubbish[np.array(permutation)]
path = np.zeros((0,2))

for i in range(len(rubbish) - 1):
    solver = Solver(points[i], points[i+1])
    opt2 =  torch.optim.Adam(solver.parameters(), lr=0.5)
    if i==9: #solver.collision().item() > 0:
        for ite in range(2500):
            y = solver.lpath() + 1e6 * solver.collision_fxy()
            if ite%100 == 0:
                print(ite, y.item())
            opt2.zero_grad()
            y.backward()
            opt2.step()
            if (ite+1)%1000 == 0:
                opt2.param_groups[0]['lr'] *= 0.5


    pp = solver.path.detach().numpy()
    path = np.concatenate([path, points[i][np.newaxis,:], pp])

path = np.concatenate([path, points[-1][np.newaxis,:]])
print(path.shape)
im = visualize(path)
cv2.imwrite('1.jpg', im)