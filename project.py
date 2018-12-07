# -*- coding: utf-8 -*-
"""
    This script computes the solution to an IVP using Runge-Kutta methods
"""
import numpy as np
import matplotlib.pyplot as plt
import math

#Hooke's values
k1 = 5000
k2 = 5000
k3 = 5000
k4 = 5000
k5 = 5000
#Floor masses
m1 = 10000
m2 = 10000
m3 = 10000
m4 = 10000
m5 = 10000
#Earthquake force
E1 = 1000
E2 = 1000
E3 = 1000
E4 = 1000
E5 = 1000

def f11(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return u12

def f12(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return -((k1+k2)*u11+k2*u21+E1)/m1

def f21(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return u22

def f22(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return (k2*u11-(k2+k3)*u21+k3*u31+E2)/m2

def f31(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return u32

def f32(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return (k3*u21-(k3+k4)*u31+k4*u41+E3)/m3

def f41(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return u42

def f42(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return (k4*u31-(k3+k4)*u41+k5*u51+E4)/m4

def f51(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return u52

def f52(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52):
    return (k5*u41-(k4+k5)*u51+E4)/m5

def rk4(n,a,b,alpha):
    """
    Given a canonical problem with f11(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52) through f52(t, u11, u12, u21, u22, u31, u32, u41, u42, u51, u52)
    and an initial condition vector on a time inteval a <= t <= b, solves using Runge-Kutta 
    fourth-order method.
    Input: 
        n  - number of points to evaluate on t-interval
        a  - initial value
        b  - terminal value
        alpha - vector of initial conditions
    
    Output:
        wxVec - vector of approximation to xth equation
        tVec - corresponding t to wxVec
    """
    h = (b-a)/n
    t = a
    w1 = alpha[0]
    w2 = alpha[1]
    w3 = alpha[2]
    w4 = alpha[3]
    w5 = alpha[4]
    w6 = alpha[5]
    w7 = alpha[6]
    w8 = alpha[7]
    w9 = alpha[8]
    w10 = alpha[9]
    
    w1Vec = [w1]
    w2Vec = [w2]
    w3Vec = [w3]
    w4Vec = [w4]
    w5Vec = [w5]
    w6Vec = [w6]
    w7Vec = [w7]
    w8Vec = [w8]
    w9Vec = [w9]
    w10Vec = [w10]
    tVec = [t]
    
    for i in range(n):
        k11 = h*f11(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k12 = h*f12(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k13 = h*f21(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k14 = h*f22(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k15 = h*f31(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k16 = h*f32(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k17 = h*f41(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k18 = h*f42(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k19 = h*f51(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        k110 = h*f52(t, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10)
        
        k21 = h*f11(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k22 = h*f12(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k23 = h*f21(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k24 = h*f22(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k25 = h*f31(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k26 = h*f32(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k27 = h*f41(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k28 = h*f42(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k29 = h*f51(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        k210 = h*f52(t+h/2, w1 + 1/2*k11, w2 + 1/2*k12, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k16, w7 + 1/2*k17, w8 + 1/2*k18, w9 + 1/2*k19, w10 + 1/2*k110)
        
        k31 = h*f11(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k23, w4 + 1/2*k24, w5 + 1/2*k25, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k32 = h*f12(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k33 = h*f21(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k34 = h*f22(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k35 = h*f31(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k36 = h*f32(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k37 = h*f41(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k38 = h*f42(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k39 = h*f51(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        k310 = h*f52(t+h/2, w1 + 1/2*k21, w2 + 1/2*k22, w3 + 1/2*k13, w4 + 1/2*k14, w5 + 1/2*k15, w6 + 1/2*k26, w7 + 1/2*k27, w8 + 1/2*k28, w9 + 1/2*k210, w10 + 1/2*k210)
        
        k41 = h*f11(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k42 = h*f12(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k43 = h*f21(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k44 = h*f22(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k45 = h*f31(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k46 = h*f32(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k47 = h*f41(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k48 = h*f42(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k49 = h*f51(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        k410 = h*f52(t+h, w1 + k31, w2 + k32, w3 + k33, w4 + k34, w5 + k35, w6 + k36, w7 + k37, w8 + k38, w9 + k39, w10 + k310)
        
        t = a + (i+1)*h 
        
        w1 = w1 + (k11 + 2*k21 + 2*k31 + k41)/6
        w2 = w2 + (k12 + 2*k22 + 2*k32 + k42)/6
        w3 = w3 + (k13 + 2*k23 + 2*k33 + k43)/6
        w4 = w4 + (k14 + 2*k24 + 2*k34 + k44)/6
        w5 = w5 + (k15 + 2*k25 + 2*k35 + k45)/6
        w6 = w6 + (k16 + 2*k26 + 2*k36 + k46)/6
        w7 = w7 + (k17 + 2*k27 + 2*k37 + k47)/6
        w8 = w8 + (k18 + 2*k28 + 2*k38 + k48)/6
        w9 = w9 + (k19 + 2*k29 + 2*k39 + k49)/6
        w10 = w10 + (k110 + 2*k210 + 2*k310 + k410)/6
        
        w1Vec.append(w1)
        w2Vec.append(w2)
        w3Vec.append(w3)
        w4Vec.append(w4)
        w5Vec.append(w5)
        w6Vec.append(w6)
        w7Vec.append(w7)
        w8Vec.append(w8)
        w9Vec.append(w9)
        w10Vec.append(w10)
        tVec.append(t)
    return w1Vec,w2Vec,w3Vec,w4Vec,w5Vec,w6Vec,w7Vec,w8Vec,w9Vec,w10Vec,tVec


# Main Code 
a = 2900 # left hand limit of t
b = 3000 # right hand limit of t
n= 10
alpha = [0, 0, 0, 0 ,0 ,0 ,0 ,0, 0, 0] # y(a) = alpha


w1Vec, w2Vec, w3Vec, w4Vec, w5Vec, w6Vec, w7Vec, w8Vec, w9Vec, w10Vec, tVec = rk4(n,a,b,alpha) # compute the rk2 approximation

n = np.array([10,20,50,100,200]) # number of points between a and b
for i in range(len(n)):
    w1Vec, w2Vec, w3Vec, w4Vec, w5Vec, w6Vec, w7Vec, w8Vec, w9Vec, w10Vec, t4Vec = rk4(n[i],a,b,alpha)


p1Flg = 1
# Plotting routine with legend for solution evolution 
if(p1Flg == 1):
    plt.figure(0)
    plt.plot(t4Vec,w1Vec,color="r", label='RK4 for equation 1')
    plt.plot(t4Vec,w2Vec,color="b", label='RK4 for equation 2')
    plt.plot(t4Vec,w3Vec,color="g", label='RK4 for equation 3')
    plt.plot(t4Vec,w4Vec,color="y", label='RK4 for equation 4')
    plt.plot(t4Vec,w5Vec,color="k", label='RK4 for equation 5')
    plt.plot(t4Vec,w6Vec,color="c", label='RK4 for equation 6')
    plt.plot(t4Vec,w7Vec,color="0.5", label='RK4 for equation 7')
    plt.plot(t4Vec,w8Vec,color="0.2", label='RK4 for equation 8')
    plt.plot(t4Vec,w9Vec,color="0.7", label='RK4 for equation 9')
    plt.plot(t4Vec,w10Vec,color="0.9", label='RK4 for equation 10')
    plt.plot()
    plt.xlabel('$t$')
    plt.ylabel('$y(t)$')
    plt.title('Approximate solution to $\dot{y}=f1(t,u1,u2)$ and $\dot{y}=f2(t,u1,u2)$')
    plt.grid(True)
    #plt.legend()
    plt.show
   # plt.savefig("proj_2_2.png")


print(rk4(1, 0, .2, alpha))
