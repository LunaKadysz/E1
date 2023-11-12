# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:32:24 2023

@author: Luna
"""

#utilidade

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def p0_(r,th):
    x = r*np.cos(th)
    y = r*np.sin(th)
    return [x,y]


def polar_to_cart(r,th):
    x = r*np.cos(th)
    y = r*np.sin(th)
    return x,y

def cart2polar(x,y):
    r = np.sqrt(x**2 + y**2)
    if x>0:
        th = np.arctan(y/x)
    elif x<0:
        th = np.arctan(y/x) + np.pi
    elif x == 0 and y > 0:
        th = np.pi/2
    elif x==0 and y<0:
        th = -np.pi/2
    return r,th

def trayectoria(X0, u, N, dt):
    """
    Calcula numéricamente la solución a dX/ds = u(X,t) cómo
    un arreglo, usando el esquema definido previamente

    Parámetros:
    X0: Condición inicial cómo arreglo de dimensión d
    u : Campo de velocidades cómo función de (x,t)
    N : Número de pasos temporales
    dt: Paso temporal
    """
    d = X0.shape[0]            # Dimensión del problema
    ts = np.arange(0, N+1)*dt  # Tiempos donde calcularé la solución
    Xs = np.zeros((N+1, d))    # Solución aproximada
    Xs[0] = X0                 # Impongo la condición inicial a t=0
    for j in range(N):
    # Aplico esquema anterior sobre Xs[j] para obtener Xs[j+1]
        #Xs[j+1] = esquema_midpoint(Xs[j], u, ts[j], dt)
        Xs[j+1] = paso_RK(Xs[j], u, ts[j], dt)
        
    return ts, Xs

def linea_de_corriente(l0, u, t, N, ds):
    """
    Calcula numéricamente la solución a dX/ds = u(X,t) cómo
    un arreglo, usando el esquema definido previamente

    Parámetros:
    l0: Origen de la línea cómo arreglo de dimensión d
    u : Campo de velocidades cómo función de (x,t)
    t : Tiempo para el cual se está calculando la línea
    N : Número de pasos en s
    ds: Paso del parametrizador
    """
    d = l0.shape[0]            # Dimensión del problema
    s = np.arange(0, N+1)*ds   # Valores del parametrizador s
    ls = np.zeros((N+1, d))    # Solución aproximada a lo largo de s
    ls[0] = l0                 # Impongo la condición inicial a s=0
    for j in range(N):
    # Aplico esquema sobre ls[j] para obtener ls[j+1]
        #ls[j+1] = esquema_midpoint(ls[j], u, t, ds)
        ls[j+1] = paso_RK(ls[j], u, t, ds)
    return s, ls


def esquema_midpoint(X, u, t, dt):
  """
  Calcula X(t+dt) a partir de X(t) y el campo de velocidades.

  Parámetros:
    X : Solución a tiempo t (X(t)) cómo arreglo de dimensión d
    u : Campo de velocidades cómo función de (x,t)
    t : Tiempo donde se conoce t
    dt: Paso temporal
  """
  return X + (dt/2)*(u(X,t)+u(X+dt*u(X,t),t+dt)) #Runge Kutta a orden 2

def paso_RK(X, u, t, dt):
    """
    Calcula X(t+dt) a partir de X(t) y el campo de velocidades.

    Parámetros:
    X : Solución a tiempo t (X(t)) cómo arreglo de dimensión d
    u : Campo de velocidades cómo función de (x,t)
    t : Tiempo donde se conoce t
    dt: Paso temporal
    """
    k1 = u(X,t)
    k2 = u(X+k1*dt/2,t)
    k3 = u(X+k2*dt/2,t)
    k4 = u(X+k3*dt,t)
    X = X + (k1 + 2*(k2+k3) + k4)/6*dt
    #t = t + dt
    return X #Runge Kutta a orden 4

def evolucionar_lineas(l0s, ts, u, N, ds, ext=None, figsize=None):
    n = l0s.shape[0]   # Cantidad de líneas de corriente
    m = ts.shape[0]    # Cantidad de tiempos (fotogramas)
    ls = np.zeros((m, n, N+1, 2))
    for i in range(m):
        for j in range(n):
            _, ls[i,j] = linea_de_corriente(l0s[j], u, ts[i], N, ds)
    xs = ls[:,:,:,0]
    ys = ls[:,:,:,1]

    if ext is None:
        extent = [np.min(np.array(xs)), np.max(np.array(xs)),
                  np.min(np.array(ys)), np.max(np.array(ys))]
    else:
        extent = ext

    # Guardo el estado de plt
    params_viejos = plt.rcParams
    plt.rc('animation', html='jshtml')

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    plt.close();  # Cerrar la figura, animation va a crear la suya propia

    # Inicializo las curvas
    plots = [ ax.plot([], [], "b-")[0] for i in range(n) ]
    dots  = [ ax.plot([], [], "ko")[0] for i in range(n) ]
    ax.set_title("$t=0$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def init():
    
        sup = 0
        for i, (x, y) in enumerate(zip(xs[0], ys[0])):
            dots[i].set_xdata([x[0]])
            dots[i].set_ydata([y[0]])
            sup = max(sup, max(np.max(np.abs(x)), np.max(np.abs(y))))

        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        return plots

    def actualizar(t):
    
        print(f"\rCalculando fotograma {t} de {m}",
              end="")

        for i, (x, y) in enumerate(zip(xs[t], ys[t])):
            plots[i].set_xdata(x)
            plots[i].set_ydata(y)

        ax.set_title(f"$t={ts[t]:.5f}$")

        return plots

        anim = animation.FuncAnimation(fig, actualizar, init_func=init,
                                     frames=range(0, m), blit=True, repeat=True)

        # Restauro el estado de plt
        plt.rc(params_viejos)

    return anim


def evolucionar_trazas(X0s, u, N, dt, paso=1, ext=None, figsize=None):
    n = X0s.shape[0]   # Cantidad de líneas de traza
    ts = np.arange(N+1)*dt
    Xs = np.zeros((N+1, n, N+1, 2))
    for j in range(n):
        Xs[:, j] = X0s[None, j, None]
    for i in range(N):
        for k in range(i//paso+1):
            Xs[i+1, j, k] = esquema(Xs[i, j, k], u, ts[i], dt)
    xs = Xs[:,:,:,0]
    ys = Xs[:,:,:,1]

    if ext is None:
        extent = [np.min(np.array(xs)), np.max(np.array(xs)),
                  np.min(np.array(ys)), np.max(np.array(ys))]
    else:
        extent = ext

    # Guardo el estado de plt
    params_viejos = plt.rcParams
    plt.rc('animation', html='jshtml')

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    plt.close();  # Cerrar la figura, animation va a crear la suya propia

    # Inicializo las curvas
    plots = [ ax.plot([], [], "b-")[0] for i in range(n) ]
    dots  = [ ax.plot([], [], "ko")[0] for i in range(n) ]
    ax.set_title("$t=0$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def init():
    
        sup = 0
        for i, (x, y) in enumerate(zip(xs[0], ys[0])):
            dots[i].set_xdata([x[0]])
            dots[i].set_ydata([y[0]])
            sup = max(sup, max(np.max(np.abs(x)), np.max(np.abs(y))))

        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        return plots

    def actualizar(t):
    
        print(f"\rCalculando fotograma {t//paso} de {(N+1)//paso}", end="")

        for i, (x, y) in enumerate(zip(xs[t], ys[t])):
            plots[i].set_xdata(x)
            plots[i].set_ydata(y)

        ax.set_title(f"$t={ts[t]:.5f}$")

        return plots

    anim = animation.FuncAnimation(fig, actualizar, init_func=init,
                                 frames=range(0, N+1, paso),
                                 blit=True, repeat=True)

  # Restauro el estado de plt
    plt.rc(params_viejos)

    return anim

