from re import S
import numpy as np
import matplotlib.pyplot as plt
from Laplacien import ResolP
from Laplacien import buildMatrix
from math import cos, sin

# Paramètres discrétisation

# Discrétisation spatiale
Nx = 40; Ny = 40 # Nx points en X / Ny points en Y
Lx = 2*np.pi; Ly = 2*np.pi # Domaine spatial
dx = Lx / Nx # Pas en x
dy = Ly / Ny # Pas en y

# Discrétsiation temporelle
Tmax = 10
dt = 1/4 * dx**2# Condition de stabilité pour schéma explicite
Nt = int(Tmax / dt)
print(Nt)

# Paramètres physiques
Re = 10
u0 = 1
Ax = 1/2
Pj = 20
lamb_x = 1/2 * Lx
Rj = Ly / 4
a = 0#2*np.pi # Pour conditions MMS

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

def getInitMMS(t):
    U = np.zeros((Nx, Ny))
    V = np.zeros((Nx, Ny))
    P = np.zeros((Nx, Ny))

    for i in range(0, Nx):
        for j in range(0, Ny):
            U[i,j] =   np.cos(Xf[i]) * np.sin(Yc[j]) * np.cos(a*t)
            V[i,j] = - np.sin(Xc[i]) * np.cos(Yf[j]) * np.cos(a*t)
            P[i,j] = - (np.cos(2*Xc[i]) + np.cos(2*Yc[j])) * np.cos(a*t)
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

def getSource(t):

    """ Renvoie les matrices des termes sources SU et SV au temps t """

    SU = np.cos(GXf)*( -np.cos(a*t)**2*np.sin(GXf) - a*np.sin(a*t)*np.sin(GYc) + 2*np.cos(a*t)*(2*np.sin(GXf) + 1/Re*np.sin(GYc)))
    SV = np.cos(GYf)*((-2/Re*np.cos(a*t) + a*np.sin(a*t))*np.sin(GXc) - (-4 + np.cos(a*t))*np.cos(a*t)*np.sin(GYf))

    return SU, SV

def getVorticity(U, V):

    """ Renvoie la vorticité du champ de vitesse (U, V) où U et V sont les matrices des composantes du champ """

    dvx = (V - np.roll(V, 1, axis = 0)) / dx # du/dx
    duy = (U - np.roll(U, 1, axis = 1)) / dy # dv/dy

    return dvx - duy

def __main__():

    # Conditions initiales
    U, V, P = getInitMMS(0)

    # Boucle temporelle
    for i in range(0, Nt):
        # On récupère les différents termes de l'équation
        GPx, GPy = getGradient(P) # Gradient de P
        LU, LV = getLinear(U, V) # Laplacien de (U, V)
        NLU, NLV = getNonLinear(U, V) # Termes non-linéaires

        SU, SV = getSource(i * dt)
        SU = np.transpose(SU)
        SV = np.transpose(SV)

        # Calculs de u* et v*
        Ustar = U + dt * (-GPx + LU - NLU + SU)
        Vstar = V + dt * (-GPy + LV - NLV + SV)

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
        #print(np.max(divUV))

        # Mise à jour de la pression
        P = P + phi

    # Pour récupérer le champ dans des fichiers textes
    np.savetxt("Data/Ufield.txt", U)
    np.savetxt("Data/Vfield.txt", V)

    # Erreur & Stabilité
    Uex, Vex, Pex = getInitMMS(Tmax)

    Ures = np.abs(Uex - U)
    Vres = np.abs(Vex - V)
    Pres = np.abs(Pex - P)

    Uerr = np.max(Ures)
    Verr = np.max(Vres)

    print(Uerr)
    print(Verr)

    # Plot des graphes
    omega = getVorticity(U, V) # Vorticité du champ à l'instant final

    plt.figure(1)

    U = np.transpose(U)
    V = np.transpose(V)

    plt.contourf(GXc, GYc, omega)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Vorticité du champ (U, V)")

    plt.figure(2)

    Uerr = np.array([0.003990961223367839, 0.0020494615290915785, 0.0002986075863157245, 9.95463685323772e-05])
    dt_steps = np.array([1/4 * dx**2, 1/4 * dx**2 / 2, 1/4 * dx**2 / 4, 1/4 * dx**2 / 6])
    plt.plot(np.log(dt_steps), np.log(Uerr))
    plt.xlabel("log(dt)"); plt.ylabel("log(max(U)) à t=Tmax")
    plt.title("Courbe de l'erreur en fonction du pas dt")

    coefs = np.polyfit(np.log(dt_steps), np.log(Uerr), 1)
    print("L'ordre temporel de la méthode est : ", coefs)

    plt.show()

__main__()
