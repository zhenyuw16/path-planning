import numpy as np 
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as sco
import torch
import torch.nn as nn

class Solver(nn.Module):
    def __init__(self, path=None):
        super(Solver, self).__init__()
        self.obstacles = [torch.tensor(  ((1.23,3.47),(1.75,4.00),(2.10,3.63),(1.58,2.30),(1.40,2.67)) ),  
                          torch.tensor(  ((4.65,5.98),(4.00,6.48),(4.52,7.68),(5.06,7.73),(5.90,6.95)) ),  
                          torch.tensor(  ((6.78,3.40),(7.78,3.76),(7.78,5.10)) ), 
                          torch.tensor(  ((4.00,3.00),(4.35,3.35),(4.80,3.45),(4.37,2.75)))  ]
        self.rubbishes = torch.tensor((  (0.40,3.00),(2.90,2.20),(1.10,6.50),(2.90,5.00),(3.00,8.30),
                            (5.20,4.80),(7.40,7.40),(5.30,1.20),(7.80,2.60),(6.00,6.00),
                            (9.00,4.80),(5.00,8.50),(7.00,1.50),(2.50,7.50)   ))
        self.car = torch.tensor([0.57/2, 0.7/2])
        #self.car = np.random.rand(2) * 10
        self.car_width = 0.57
        self.car_length = 0.7
        self.path =  nn.Parameter(path,  requires_grad=True) if path is not None else nn.Parameter(torch.tensor(np.random.rand(99,2).astype('float32') * 10,  requires_grad=True))
    
    def rectangle(self, p):
        r = torch.tensor([[p[0]-self.car_width/2, p[1]-self.car_length/2], 
                      [p[0]+self.car_width/2, p[1]-self.car_length/2], 
                      [p[0]+self.car_width/2, p[1]+self.car_length/2], 
                      [p[0]-self.car_width/2, p[1]+self.car_length/2], 
                    ])
        return r

    def visualize(self):
        im = np.zeros((1000,1000), dtype = np.uint8)
        r = self.rectangle(self.car)
        point_obstacles = []
        for i in range(len(self.obstacles)):
            p = self.obstacles[i].numpy()
            p[:,0] = p[:,0]
            p[:,1] = 10 - p[:,1]
            point_obstacles.append(p)
        rr = r.numpy()
        rr[:,0] = r[:,0]
        rr[:,1] = 10 - r[:,1]
        for i in range(len(point_obstacles)):
            cv2.fillConvexPoly(im, (point_obstacles[i]*100).astype('int'), 255)
        #cv2.fillConvexPoly(im, (rr*100).astype('int'), 255)
        for i in range(len(self.rubbishes)):
            cv2.circle(im, (int(self.rubbishes[i][0]*100), int(1000 - self.rubbishes[i][1]*100)), 5, 255, -1)
        pp = self.path.detach().numpy().copy()
        pp[:,1] = 10 - pp[:,1]
        cv2.polylines(im, np.array([(pp*100)]).astype('int'), False, 128, 3)
        return im


    def collision(self, path=None):
        #r = self.rectangle()
        #path = torch.cat([self.car[np.newaxis,:], torch.clamp(self.path, 0, 10)], 0)
        #path = torch.sigmoid(self.path) * 10
        path = self.path if path is None else path
        #path =  torch.clamp(self.path, 0, 10)
        path = torch.cat([self.car[np.newaxis,:], path], 0)
        p = path[0:-1]
        r = path[1:] - path[0:-1]
        colllist = []
        
        for i in range(len(self.obstacles)):
            #print(linem, lineb)
            ob = self.obstacles[i]
            sob = ob.shape[0]
            q = ob[:,:]
            s = ob[list(range(1, sob)) + [0]] - q
            qp = (q[np.newaxis,:,:]-p[:,np.newaxis,:])
            t = (qp[:,:,0] * s[np.newaxis,:,1] - qp[:,:,1] * s[np.newaxis,:,0]) / ( torch.transpose((r[np.newaxis,:,0] * s[:,np.newaxis,1] - r[np.newaxis,:,1] * s[:,np.newaxis,0]), 1, 0) + 1e-8)
            u = (qp[:,:,0] * r[:,np.newaxis,1] - qp[:,:,1] * r[:,np.newaxis,0]) / ( torch.transpose((r[np.newaxis,:,0] * s[:,np.newaxis,1] - r[np.newaxis,:,1] * s[:,np.newaxis,0]), 1, 0) + 1e-8)
            #print(torch.sum((t>0)& (t <1)), torch.sum((u>0)& (u<1)))
            #collt = 1 - (torch.sign(t) * torch.sign(t - 1/2) + 1) / 2
            #collu = 1 - (torch.sign(u) * torch.sign(u - 1/2) + 1) / 2
            collt = 1 - (t/torch.abs(t) * (t-1)/torch.abs(t-1) + 1) / 2
            collu = 1 - (u/torch.abs(u) * (u-1)/torch.abs(u-1) + 1)  / 2
            coll = collt * collu
            colllist.append(coll)
        
        colllist = torch.cat(colllist, 1)
        return torch.sum(colllist) #torch.sum(coll * (1 - coll) ) #- coll * torch.log(coll+1e-11))
    
    def lpath(self):
        #path = np.reshape(path, (-1,2))
        #path =  torch.clamp(self.path, 0, 10)
        #path = torch.sigmoid(self.path) * 10
        path = self.path
        path = torch.cat([self.car[np.newaxis,:], path], 0)
        xp = path[1:] - path[0:-1]
        ll = torch.sum(torch.sqrt(1e-6 + torch.sum(xp**2, 1)))
        return ll
    
    def clean(self):
        #path =  torch.clamp(self.path, 0, 10)
        #path = torch.sigmoid(self.path) * 10
        path = self.path
        dis = path[:,np.newaxis,:] - self.rubbishes[np.newaxis,:,:]
        dis = torch.sum(dis**2, 2)
        dis = torch.min(dis, 0)[0]
        return torch.sum(dis)



solver = Solver()
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
opt2 =  torch.optim.Adam(solver.parameters(), lr=0.05) # torch.optim.SGD(solver.parameters(), lr=0.1) #


gclip = 1


for i in range(15000):
    #print(solver.collision())
    y = solver.collision() + solver.lpath() + 1e5 * solver.clean() 
    if i%100 == 0:
        print(i, y.item())
    
    opt2.zero_grad()
    y.backward()
    for group in opt2.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-gclip, gclip)

    opt2.step()
    #solver.path = torch.clamp(solver.path, 0, 10)
    if (i+1)%2500 == 0:
        gclip *= 0.5
        opt2.param_groups[0]['lr'] *= 0.5
    
    if np.max(solver.path.detach().numpy()) > 10 or np.min(solver.path.detach().numpy()) < 0:
        break
    




#print(solver.path.detach().numpy())
import pickle
#pickle.dump([x,g,yy], open('a.pkl','wb'))
im = solver.visualize()
cv2.imwrite('1.jpg', im)