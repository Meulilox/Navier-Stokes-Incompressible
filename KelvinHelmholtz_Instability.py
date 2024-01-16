import numpy as np
import matplotlib.pyplot as plt
from math import *
from Laplacien import ResolP
from Laplacien import buildMatrix
from matplotlib import cm

# Paramètres discrétisation

# Discrétisation spatiale
Nx = 30; Ny = 30 # Nx points en X / Ny points en Y
Lx = 2; Ly = 1 # Domaine spatial
dx = Lx / Nx # Pas en x
dy = Ly / Ny # Pas en y

# Discrétsiation temporelle
Tmax = 1.05
dt = 1/4 * dx**2 # Condition de stabilité pour schéma explicite
Nt = int(Tmax / dt)

# Paramètres physiques
Re = 1000
u0 = 1
Ax = 1/2
Pj = 20
lamb_x = 1/2 * Lx
Rj = Ly / 4

# Maillages

# Maillages "faces"
Xf = np.linspace(0, Lx, Nx, endpoint = False)
Yf = np.linspace(0, Ly, Ny, endpoint = False)

# Maillages "centres"
Xc = np.linspace(dx/2, Lx-dx/2, Nx)
Yc = np.linspace(dy/2, Ly-dy/2, Ny)

# Meshgrids
GXc, GYc = np.meshgrid(Xc, Yc)
GXf, GYf = np.meshgrid(Xf, Yf)

def getInit():
    """ Renvoie les conditions initiales des champs (U, V) et P """
    U = np.zeros((Nx, Ny))
    V = np.zeros((Nx, Ny))
    P = np.zeros((Nx, Ny))

    for i in range(0, Nx):
        for j in range(0, Ny):
            # Maillage pour U : Xf et Yc
            u1 = u0/2 * (1 + np.tanh(Pj / 2 * (1 - np.abs(Ly / 2 - Yc[j])/Rj)))
            u2 = Ax * np.sin(2*np.pi*Xf[i]/lamb_x)
            U[i, j] = u1 * (1 + u2)

    return U, V, P

def getGradient(S):
    """ Renvoie les matrices Gx et Gy du gradient de S

    Args :
        S(np.array()) : Matrice
    """
    # 1 décale vers la droite donc taux d'accroissement pris à gauche
    Gx = (S - np.roll(S, 1, axis = 0)) / dx # dS/dx
    Gy = (S - np.roll(S, 1, axis = 1)) / dy # dS/dy

    return Gx, Gy

def getDiv(A, B):
    """ Renvoie la matrice de la divergence du champ vectoriel (A, B) """
    # -1 décale vers la gauche donc taux d'accroissement pris à droite
    divX = (np.roll(A, -1, axis = 0) - A) / dx # dA/dx
    divY = (np.roll(B, -1, axis = 1) - B) / dy # dB/dy

    return divX + divY

def getLinear(U, V):
    """ Renvoie les termes linéaires LU et LV (laplacien(u, v) dans N-S) """

    LUx = (np.roll(U, 1, axis = 0) - 2*U + np.roll(U, -1, axis = 0)) / dx**2 # d^2u/dx**2
    LUy = (np.roll(U, 1, axis = 1) - 2*U + np.roll(U, -1, axis = 1)) / dy**2 # d^2u/dy**2
    LU = LUx + LUy # lap(u)

    LVx = (np.roll(V, 1, axis = 0) - 2*V + np.roll(V, -1, axis = 0)) / dx**2 # d^2v/dx**2
    LVy = (np.roll(V, 1, axis = 1) - 2*V + np.roll(V, -1, axis = 1)) / dy**2 # d^2v/dy**2
    LV = LVx + LVy # lap(v)

    LU = LU / Re
    LV = LV / Re

    return LU, LV

def getNonLinear(U, V):
    """ Renvoie les matrices des termes convectifs : NLu et NLv """

    UD = (U + np.roll(U, -1, axis = 0)) / 2
    UG = (U + np.roll(U,  1, axis = 0)) / 2
    UH = (U + np.roll(U, -1, axis = 1)) / 2
    UB = (U + np.roll(U,  1, axis = 1)) / 2

    VD = (V + np.roll(V, -1, axis = 0)) / 2
    VG = (V + np.roll(V,  1, axis = 0)) / 2
    VH = (V + np.roll(V, -1, axis = 1)) / 2
    VB = (V + np.roll(V,  1, axis = 1)) / 2

    # Valeurs de V aux noeuds de U
    V_povU = 1/4 * (V + np.roll(V, -1, axis = 1) +\
                        np.roll(V,  1, axis = 0) +\
                        np.roll(V, [1, -1], axis = (0, 1)))

    # Valeurs de U aux noeuds de V
    U_povV = 1/4 * (U + np.roll(U, -1, axis = 0) +\
                        np.roll(U,  1, axis = 1) +\
                        np.roll(U, [-1, 1], axis = (0, 1)))

    # NLu
    NLU_1 = U      * (UD - UG) / dx
    NLU_2 = V_povU * (UH - UB) / dy
    NLU = NLU_1 + NLU_2

    # NLv
    NLV_1 = U_povV * (VD - VG) / dx
    NLV_2 = V      * (VH - VB) / dy
    NLV = NLV_1 + NLV_2

    return NLU, NLV

def getVorticity(U, V):
    """ Pour récupérer la matrice du champ scalaire de vorticité """

    dvx = (V - np.roll(V, 1, axis = 0)) / dx # du/dx
    duy = (U - np.roll(U, 1, axis = 1)) / dy # dv/dy

    return dvx - duy

def toMatrix(U):
    """ Pour mettre sous forme matricielle un champ stocké dans un vecteur """
    res = np.zeros((Nx, Ny))
    for p in range(0, Nx*Ny):
        i = p % Nx
        j = p // Nx
        res[i, j] = U[p]

    return res

def toVector(U):
    """ Pour mettre sous forme vectorielle un champ stocké dans une matrice """
    res = np.zeros((Nx*Ny))

    for i in range(0, Nx):
        for j in range(0, Ny):
            res[i+j*Nx] = U[i,j]

    return res

def __main__():

    # Conditions initiales
    U, V, P = getInit()

    # Boucle temporelle
    for i in range(0, Nt):
        # On récupère les différents termes de l'équation
        GPx, GPy = getGradient(P) # Gradient de P
        LU, LV = getLinear(U, V) # Laplacien de (U, V)
        NLU, NLV = getNonLinear(U, V) # Termes non-linéaires

        # Calculs de u* et v* (Euler explicite)
        """Ustar = U + dt * (-GPx + LU - NLU)
        Vstar = V + dt * (-GPy + LV - NLV)"""

        # Calculs de u* et v* (Schéma implicite pour les termes linéaires)
        # On met tous les termes sous forme de vecteurs
        """GPx = toVector(GPx); GPy = toVector(GPy)
        LU = toVector(LU); LV = toVector(LV)
        NLU = toVector(NLU); NLV = toVector(NLV)
        U = toVector(U); V = toVector(V)

        # Matrice à inverser pour les termes linéaires en implicite
        M = buildMatrix(Lx, Ly, Nx, Ny)
        A = np.eye(Nx*Ny) - dt/Re*M

        # Calcul u* et v*
        Ustar = np.linalg.solve(A, U + dt * (-GPx - NLU))
        Vstar = np.linalg.solve(A, V + dt * (-GPy - NLV))

        # On remet tout sous forme de matrice
        GPx = toMatrix(GPx); GPy = toMatrix(GPy)
        LU = toMatrix(LU); LV = toMatrix(LV)
        NLU = toMatrix(NLU); NLV = toMatrix(NLV)
        Ustar = toMatrix(Ustar); Vstar = toMatrix(Vstar)"""

        # Calculs de u* et v* (RK2)
        # k1 correspond à -GPx + LU - NLU
        # k2 correspond à -GPx + LU' - NLU' où LU' et NLU' sont les termes linéaires et non-linéaires pour les champs U + dt*U_k1, V + dt*V_k1

        '''U_k1 = -GPx + LU - NLU
        V_k1 = -GPy + LV - NLV

        LUp, LVp = getLinear(U + dt*U_k1, V + dt*V_k1) # Laplacien de (U, V)
        NLUp, NLVp = getNonLinear(U + dt*U_k1, V + dt*V_k1) # Termes non-linéaires

        U_k2 = -GPx + LUp - NLUp
        V_k2 = -GPy + LVp - NLVp

        Ustar = U + dt/2 * (U_k1 + U_k2)
        Vstar = V + dt/2 * (V_k1 + V_k2)'''

        # Calculs de u* et v* (RK4) (calculs plus long)

        U_k1 = -GPx + LU - NLU
        V_k1 = -GPy + LV - NLV

        LUk1, LVk1 = getLinear(U + dt/2*U_k1, V + dt/2*V_k1)
        NLUk1 , NLVk1 = getNonLinear(U + dt/2*U_k1, V + dt/2*V_k1)

        U_k2 = -GPx + LUk1 + NLUk1
        V_k2 = -GPy + LVk1 + NLVk1

        LUk2, LVk2 = getLinear(U + dt/2*U_k2, V + dt/2*V_k2)
        NLUk2 , NLVk2 = getNonLinear(U + dt/2*U_k2, V + dt/2*V_k2)

        U_k3 = -GPx + LUk2 + NLUk2
        V_k3 = -GPy + LVk2 + NLVk2

        LUk3, LVk3 = getLinear(U + dt/2*U_k3, V + dt/2*V_k3)
        NLUk3 , NLVk3 = getNonLinear(U + dt/2*U_k3, V + dt/2*V_k3)

        U_k4 = -GPx + LUk3 + NLUk3
        V_k4 = -GPy + LVk3 + NLVk3

        Ustar = U + dt * (U_k1/6 + U_k2/3 + U_k3/3 + U_k4/6)
        Vstar = V + dt * (V_k1/6 + V_k2/3 + V_k3/3 + V_k4/6)

        # Calcul de la divergence du champ (u*, v*)
        Div_UVstar = getDiv(Ustar, Vstar) / dt

        # Calcul de phi
        phi = ResolP(Lx, Ly, Nx, Ny, Div_UVstar) # A compléter
        GphiX, GphiY = getGradient(phi)

        # Pas de correction
        U = Ustar - dt * GphiX
        V = Vstar - dt * GphiY

        # Divergence de (U, V)
        divUV = getDiv(U, V)
        print(np.max(divUV))

        # Mise à jour de la pression
        P = P + phi

        # Pour arrêter la simulation à 1/3 * Tmax ou 2/3 * Tmax
        """if (i > Nt/3):
            break"""

    # On sauvegarde les solutions dans un fichier texte
    #np.savetxt("Data/Ufield.txt", U)
    #np.savetxt("Data/Vfield.txt", V)

    # Plot des graphes
    omega = getVorticity(U, V) # Vorticité du champ à l'instant final
    omega = np.transpose(omega)

    plt.figure(1)
    """ax = plt.axes(projection='3d')
    ax.plot_surface(GX, GY, omega, cmap = cm.viridis, linewidth=0, antialiased=True)
    ax.set_title("Vorticité")
    ax.set_xlabel("X"); ax.set_ylabel("Y")"""
    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Vorticité pour Nx = Ny = 30")
    plt.contourf(GXc, GYc, omega)

    plt.show()

__main__()
