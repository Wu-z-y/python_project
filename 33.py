#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def create_Ray(O, D):
    ray={'orig':np.array(O),
         'dir':np.array(D)}
    return ray

def create_sphere(P, r, a, d, s, ri, i):
    sphere={'type':'sphere',
            'position':np.array(P),
            'radius':r,
            'ambient':np.array(a),
            'diffuse' : np.array(d),
            'specular' : np.array(s),
            'reflection index':np.array(ri),
            'index':i}
    return sphere

def create_plane(P, n,a,d,s,ri,i):
    plane={'type':'plane',
           'position':np.array(P),
           'normal':np.array(n),
            'ambient':np.array(a),
            'diffuse' : np.array(d),
            'specular' : np.array(s),
            'reflection index':np.array(ri),
           'index':i}
    return plane

def normalize(x):
    return x/np.linalg.norm(x) 

def rayAt(ray,t):
    D=ray['dir']
    O=ray['orig']
    return np.array(O+t*D)
        

def intersect_Plane(ray, plane):
    d=ray['dir']
    O=ray['orig']
    P=plane['position']
    n=plane['normal']
    denom = np.dot(d, n)
    if abs(denom)<=10**(-6):
        return np.inf
    else : 
        t=-(np.dot((O-P),n)/denom)
        if t>0:
            return t
        else : 
            return np.inf

def intersect_Sphere(ray, sphere):
    d=ray['dir']
    O=ray['orig']
    P=sphere['position']
    r=sphere['radius']
    a = np.linalg.norm(d)**2
    b = 2*np.dot(O-P,d)
    c = np.linalg.norm(O-P)**2-r**2
    delta = b**2-4*a*c
    if delta > 0:
        t1 = (-b-np.sqrt(delta))/(2*a)
        t2 = (-b+np.sqrt(delta))/(2*a)
        if t1 > 0 and t2 > 0 :
            return min(t1,t2)
        else : 
            return np.inf 
    else : 
        return np.inf
    

def get_Normal(obj, M):
    if obj['type']=='sphere':
        N= normalize(M-obj['position'])
    elif obj['type']=='plane':
        N= obj['normal']
    return N


def intersect_Scene(ray, obj):
    if obj['type'] == 'sphere': 
        return intersect_Sphere(ray, obj)
    elif obj['type'] == 'plane': 
        return intersect_Plane(ray, obj)

def Is_in_Shadow(obj_min,P,N):
    l=[]
    lpl= normalize(Light['position']-P)
    PE = P + np.multiply(N,acne_eps)
    rayTest = create_Ray(PE, lpl)
    for obj_test in scene:
        if (obj_test['index'] != obj_min['index']):
            inter = intersect_Scene(rayTest,obj_test)
            if (inter != np.inf):
                l=l+[obj_test]
    if len(l)==0:
        return True
    else:
        return False

def eclairage(obj,light,P) : 
    # Remplissez ici 
    Ka = obj['ambient']
    la = light['ambient']
    Kd=obj['diffuse']
    ld=light['diffuse']
    Ks=obj['specular']
    ls=light['specular']
    n = get_Normal(obj, P)
    L=light['position']
    l = normalize(L-P)
    c = normalize(C-P)
    lc = normalize(l+c)
    ct= Ka*la+Kd*ld*max(np.dot(l,n),0)+Ks*ls*max(np.dot(lc,n),0)**(materialShininess/4)
    return ct

def reflected_ray(dirRay,N):
    d=dirRay['dir']
    d0=d-2*(np.dot(d,N)/np.linalg.norm(d)*np.linalg.norm(N))*N
    return d0

def compute_reflection(rayTest,depth_max,col):
    d=rayTest['dir']
    c=1
    for k in range(0, depth_max):
        trac= trace_ray(rayTest)
        if(trac==None):
            break
        obj,M,N,col_ray = trac
        col = col+ c*col_ray
        Me= M+N*acne_eps
        dirRay= create_Ray(Me,d)
        d= reflected_ray(dirRay,N)
        rayTest= create_Ray(Me,d)
        c= c*obj['reflection index']
    return col 

def trace_ray(ray):
    # Remplissez ici 
    tMin= np.inf
    objMin= None
    for e in scene : 
        t = intersect_Scene(ray, e)
        if t<tMin: 
            objMin=e
            tMin=t
    
    if objMin==None:
        return None
    P=rayAt(ray,tMin)
    n=get_Normal(objMin,P)
    if not Is_in_Shadow(objMin,P,n):
        return None
    col= eclairage(objMin,Light,P)
    return objMin,P,n,col


# Taille de l'image
w = 800
h = 600
acne_eps = 1e-4
materialShininess = 50


img = np.zeros((h, w, 3)) # image vide : que du noir
#Aspect ratio
r = float(w) / h
# coordonnées de l'écran : x0, y0, x1, y1.
S = (-1., -1. / r , 1., 1. / r )


# Position et couleur de la source lumineuse
Light = { 'position': np.array([5, 5, 0]),
          'ambient': np.array([0.05, 0.05, 0.05]),
          'diffuse': np.array([1, 1, 1]),
          'specular': np.array([1, 1, 1]) }

L = Light['position']


col = np.array([0.2, 0.2, 0.7])  # couleur de base
C = np.array([0., 0.1, 1.1])  # Coordonée du centre de la camera.
Q = np.array([0,0.3,0])  # Orientation de la caméra
img = np.zeros((h, w, 3)) # image vide : que du noir
materialShininess = 50
skyColor = np.array([0.321, 0.752, 0.850])
whiteColor = np.array([1,1,1])
depth_max = 10

scene = [create_sphere([.75, -.3, -1.], # Position
                         .6, # Rayon
                         np.array([1. , 0.6, 0. ]), # ambiant
                         np.array([1. , 0.6, 0. ]), # diffuse
                         np.array([1, 1, 1]), # specular
                         0.2, # reflection index
                         1), # index
          create_plane([0., -.9, 0.], # Position
                         [0, 1, 0], # Normal
                         np.array([0.145, 0.584, 0.854]), # ambiant
                         np.array([0.145, 0.584, 0.854]), # diffuse
                         np.array([1, 1, 1]), # specular
                         0.7, # reflection index
                         2), # index
         ]

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col = np.array([0,0,0])
        M=np.array([x, y, 0])
        d=normalize(M-C)
        raytest=create_Ray(C, d)
        resultat=trace_ray(raytest)
        if resultat!=None:
             obj,P,n,colray=resultat
             col =col + compute_reflection(raytest,depth_max,col)
        img[h - j - 1, i, :] = np.clip(col, 0, 1) # la fonction clip permet de "forcer" col a être dans [0,1]

plt.imsave('figRaytracing.png', img)