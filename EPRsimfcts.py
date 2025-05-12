import numpy as np
from numba import njit, jit, prange

#Note: this function library depends on Numba and Numpy to run.

@njit
def T10S(a):
    mat=a*np.array([[-1, 0, 0],[0,0,0],[0,0,1]])
    return mat

@njit
def T1pS(a):
    mat=a*np.array([[0, 0, 0],[-1,0,0],[0,-1,0]])
    return mat

@njit
def T1mS(a):
    mat=a*np.array([[0, 1, 0],[0,0,1],[0,0,0]])
    return mat

@njit
def Dij1(phi, theta, chi, i, j): #Rotational matrix D^1(Omega)*
    Drot=np.array([[(1+np.cos(theta))/2*np.exp(1j*chi+1j*phi),  -np.sin(theta)/np.sqrt(2)*np.exp(1j*chi), (1-np.cos(theta))/2*np.exp(1j*chi-1j*phi)],
                   [np.sin(theta)/np.sqrt(2)*np.exp(1j*phi),                np.cos(theta),                -np.sin(theta)/np.sqrt(2)*np.exp(-1j*phi)],
                   [(1-np.cos(theta))/2*np.exp(-1j*chi+1j*phi), np.sin(theta)/np.sqrt(2)*np.exp(-1j*chi), (1+np.cos(theta))/2*np.exp(-1j*chi-1j*phi)]])
    if int(i)!=1 and int(i)!=0 and int(i)!=-1:
        return 0
    elif int(j)!=1 and int(j)!=0 and int(j)!=-1:
        return 0
    else:
        return Drot[int(j+1), int(i+1)]

gamma_e=28.02495142 #Unit: 2pi X GHz/T
@njit
def Hzee(B0, phi, theta, chi):
    return gamma_e*B0*(T1pS(Dij1(phi, theta, chi, 0, -1))+T10S(Dij1(phi, theta, chi, 0, 0))+T1mS(Dij1(phi, theta, chi, 0, 1)))

@njit
def Hmol(D, E):
    return np.array([[D/3, 0, E],[0, -2*D/3, 0],[E, 0, D/3]])

@njit
def H0(D, E, B0, phi, theta, chi):
    return Hzee(B0, phi, theta, chi)+Hmol(D, E)

@njit
def p_function(B_0, D_0, E_0):
    p = -(gamma_e**2*B_0**2 + E_0**2 + D_0**2 / 3)
    return p
@njit    
def q_function(B_0, D_0, E_0, theta, chi):   
    q = (2 * D_0**3 / 27) - (2*D_0 * E_0**2 / 3) - (2 * gamma_e**2*B_0**2*D_0 / 3) * (3*np.cos(theta)**2-1)/2 - gamma_e**2*B_0**2 * E_0*np.cos(2*chi)*np.sin(theta)**2
    return q
    
@njit
def ang(p,q):
    angle = (1/3) * np.arccos(3*q/(2*p)*np.sqrt(-3/p))
    return angle

@njit
def Energies_sorted(p, angle):
    E_k = np.zeros(3)
    for k in range(3):
        E_k[k] = (2 * np.sqrt(-p / 3) * np.cos(angle+2*np.pi*(k+1)/3))
    return np.sort(E_k) 

@njit
def Energies_unsorted(p, angle):
    E_k = np.zeros(3)
    for k in range(3):
        E_k[k] = (2 * np.sqrt(-p / 3) * np.cos(angle+2*np.pi*(k+1)/3))
    return E_k

@jit(nopython=True)
def BinCoef(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    c = 1
    for i in range(int(k)):
        c = c * (n - i) / (i + 1)
    return c

BCList=[]
for n in range(300):
    for k in range(n+1):
        BCList.append(BinCoef(n, k))
BCList=np.array(BCList)

@jit(nopython=True)
def BCidx(n, k):
    return(int(n*(n+1)/2+k))

@jit(nopython=True)
def BC_read_from_list(n, k):
    return BCList[BCidx(n, k)]

@jit(nopython=True)
def cg_bc(J1, J2, J, m1, m2, m):
    delta = (m == (m1 + m2)) 
    if np.abs(m)>J or np.abs(m1)>J1 or np.abs(m2)>J2 or J1+J2<J or J1+J<J2 or J2+J<J1 or (not delta):
        return 0
    else:
        A=(2*J+1)**2 /(2*J1+1) /(2*J2+1) * BC_read_from_list(J1+J2+J+1, J1+J2-J) * BC_read_from_list(2*J, J+m)/BC_read_from_list(J1+J2+J+1, J1-J2+J)/BC_read_from_list(J1+J2+J+1, J2-J1+J)/BC_read_from_list(2*J1, J1+m1) /BC_read_from_list(2*J2, J2+m2) 
        sumlist=np.array([(-1)**t * BC_read_from_list(J1+J2-J, t) *BC_read_from_list(J-m, J1-m1-t) * BC_read_from_list(J+m, J2+m2-t)        for t in range(int(max(0, J1-m1-J+m, J2+m2-J-m)), 1+int(min(J1-m1, J2+m2, J1+J2-J)))])
        Res=delta*np.sqrt(A)*sum(sumlist)
        if np.isnan(Res):
            Res=0
    return Res

@jit(nopython=True)
def w3j_bc(j1, j2, j, m1, m2, m):
    if m1+m2+m!=0:
        return 0
    return (-1)**(j1-j2-m)/np.sqrt(2*j+1)*cg_bc(j1, j2, j, m1, m2, -m)

#Spherical harmonics for up to l=2
@njit
def SH(l,m,theta, phi):
    Ylm=0+0*1j
    if l==0:
        if m ==0:
            Ylm=np.sqrt(1/np.pi)/2
    elif l==1:
        if m==-1:
            Ylm = 0.5*np.sqrt(1.5/np.pi)*np.sin(theta)*np.exp(-1j*phi)
        elif m==0:
            Ylm = 0.5*np.sqrt(3/np.pi)*np.cos(theta)
        elif m==1:
            Ylm = -0.5*np.sqrt(1.5/np.pi)*np.sin(theta)*np.exp(1j*phi)
    elif l==2:
        if m==-2:
            Ylm = 0.25*np.sqrt(7.5/np.pi)*np.sin(theta)**2*np.exp(-2*1j*phi)
        elif m==-1:
            Ylm = 0.5*np.sqrt(7.5/np.pi)*np.sin(theta)*np.cos(theta)*np.exp(-1j*phi)
        elif m==0:
            Ylm = 0.25*np.sqrt(5/np.pi)*(3*np.cos(theta)**2-1)
        elif m==1:
            Ylm = -0.5*np.sqrt(7.5/np.pi)*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
        elif m==2:
            Ylm = 0.25*np.sqrt(7.5/np.pi)*np.sin(theta)**2*np.exp(2*1j*phi)
    return Ylm

@njit
def D_cc(p, q, chi, theta, phi):#Rank-1 Wigner D matrix complex conjungate, D complex conjungate
    phase=np.exp(-1j*q*chi-1j*p*phi)
    d_theta = 1/2
    if (p+q) % 2 !=0:
        if p<q:
            d_theta = np.sin(theta)/np.sqrt(2)
        else:
            d_theta = -np.sin(theta)/np.sqrt(2)
    else:
        if p==0 and q==0:
            d_theta = np.cos(theta)
        elif p==q:
            d_theta = d_theta + np.cos(theta)/2
        else:
            d_theta = d_theta - np.cos(theta)/2
    return phase*d_theta

@njit
def C_q(q1, theta_01, chi_01, theta_12, phi_12):
    c=-4*np.sqrt(2)*np.pi #coef before the summation
    s=0+0j
    for q2 in [-1, 0, 1]:
        q=q1+q2
        w3j = w3j_bc(1, 1,2, q1, q2, -q)
        s+=w3j * np.conj(SH(1, q2, theta_01, chi_01))*SH(2, -q, theta_12, phi_12)
    return s*c
    
def population_vs_orientation(nx0,ny0, nz0, B, D, E, phi, theta, chi):
    H_0 = H0(D, E, 0, phi, theta, chi)
    H_1 = H0(D, E, B, phi, theta, chi)
    E_0, M0=np.linalg.eigh(H_0)
    E_1, M1=np.linalg.eigh(H_1)
    nx=nx0*np.abs(np.matmul(M0.T[2], M1.T[2]))**2+ny0*np.abs(np.matmul(M0.T[1], M1.T[2]))**2+nz0*np.abs(np.matmul(M0.T[0], M1.T[2]))**2
    ny=nx0*np.abs(np.matmul(M0.T[2], M1.T[1]))**2+ny0*np.abs(np.matmul(M0.T[1], M1.T[1]))**2+nz0*np.abs(np.matmul(M0.T[0], M1.T[1]))**2
    nz=nx0*np.abs(np.matmul(M0.T[2], M1.T[0]))**2+ny0*np.abs(np.matmul(M0.T[1], M1.T[0]))**2+nz0*np.abs(np.matmul(M0.T[0], M1.T[0]))**2
    return nx, ny, nz

#@njit
def Rabi_freq_vs_orientation(B0, B1, phi, theta, chi):
    Drot=np.array([[(1+np.cos(theta))/2*np.exp(-1j*chi-1j*phi),-np.sin(theta)/np.sqrt(2)*np.exp(-1j*chi) , (1-np.cos(theta))/2*np.exp(-1j*chi+1j*phi)],
                   [np.sin(theta)/np.sqrt(2)*np.exp(-1j*phi), np.cos(theta),-np.sin(theta)/np.sqrt(2)*np.exp(1j*phi) ],
                   [(1-np.cos(theta))/2*np.exp(1j*chi-1j*phi),np.sin(theta)/np.sqrt(2)*np.exp(1j*chi),(1+np.cos(theta))/2*np.exp(1j*chi+1j*phi)]])
    
    H_1 = H0(2.356, 0.458, B0, 0, theta, chi)
    E_1, M1=np.linalg.eigh(H_1)
    Eigvec=M1.T #Here we may just inherit the eigen-state vectors from the main program to reduce the time of calculation
    #Note: we need to check if Tz and Tx/Ty energy curves have crossing so that we are not driving a wrong transition
    Txvec=Eigvec[2]
    Tzvec=Eigvec[0]
    TxTmqTz=np.matmul(Txvec.conj(), np.matmul(T1mS(Drot[2, 2]-Drot[0, 2]), Tzvec))
    TxT0qTz=np.matmul(Txvec.conj(), np.matmul(T10S(Drot[2, 1]-Drot[0, 1]), Tzvec))
    TxTpqTz=np.matmul(Txvec.conj(), np.matmul(T1pS(Drot[2, 0]-Drot[0, 0]), Tzvec))
    sum_over_q=-TxTmqTz+TxT0qTz-TxTpqTz
    if np.abs(sum_over_q)<1e-10 and B0>0:
        Txvec=Eigvec[2]
        Tzvec=Eigvec[1]
        TxTmqTz=np.matmul(Txvec.conj(), np.matmul(T1mS(Drot[2, 2]-Drot[0, 2]), Tzvec))
        TxT0qTz=np.matmul(Txvec.conj(), np.matmul(T10S(Drot[2, 1]-Drot[0, 1]), Tzvec))
        TxTpqTz=np.matmul(Txvec.conj(), np.matmul(T1pS(Drot[2, 0]-Drot[0, 0]), Tzvec))
        sum_over_q=-TxTmqTz+TxT0qTz-TxTpqTz
        
    Omega_1=2*np.pi*28024.9514242*B1*sum_over_q/np.sqrt(2) #unit: MHz/T
    
    return np.abs(Omega_1)


@njit
def Eff_H(Omega: complex, delta: complex)-> complex:
    return np.array([[delta, Omega],[Omega.conjugate(), -delta]])/2

I2 = np.eye(2, dtype=np.complex128)

sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def expm_hamiltonian(H, t):
    """
    Compute the time evolution operator U(t) = exp(-i H t) for a 2x2 Hermitian matrix H,
    using its decomposition into the identity and Pauli matrices.
    
    Parameters:
      H : 2x2 numpy array (Hermitian)
      t : time at which to evaluate U(t)
    
    Returns:
      U : 2x2 numpy array representing the time evolution operator at time t
    """
    # Decompose H = a0 * I + a_vec · sigma.
    # a0 is half the trace of H:
    a0 = 0.5 * np.trace(H)
    
    # Extract the coefficients for the Pauli matrices:
    # a_x = 0.5 * Re(H[0,1] + H[1,0])
    ax = 0.5 * (H[0, 1] + H[1, 0]).real
    # a_y = -0.5 * Im(H[0,1] - H[1,0])
    ay = -0.5 * (H[0, 1] - H[1, 0]).imag
    # a_z = 0.5 * (H[0,0] - H[1,1])
    az = 0.5 * (H[0, 0] - H[1, 1]).real
    a_vec = np.array([ax, ay, az])
    
    # Compute the norm of the vector part
    a_norm = np.linalg.norm(a_vec)
    
    # Overall phase factor from the trace part
    phase = np.exp(-1j * a0 * t)
    
    # If a_norm is nonzero, construct n_dot_sigma, otherwise just return phase*I.
    if a_norm > 1e-12:
        # Build the operator n·sigma using the Pauli matrices.
        n_dot_sigma = (ax * sx + ay * sy + az * sz) / a_norm
        U = phase * ( np.cos(a_norm * t) * I2 - 1j * np.sin(a_norm * t) * n_dot_sigma )
    else:
        U = phase * I2
        
    return U
    
@njit
def expm_hamiltonian_numba(H, t):
    a0 = 0.5 * (H[0, 0] + H[1, 1]).real
    ax = 0.5 * (H[0, 1] + H[1, 0]).real
    ay = -0.5 * (H[0, 1] - H[1, 0]).imag
    az = 0.5 * (H[0, 0] - H[1, 1]).real
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)

    # Compute phase manually
    phase_real = np.cos(a0 * t)
    phase_imag = -np.sin(a0 * t)

    U = np.zeros((2, 2), dtype=np.complex128)
    
    if a_norm > 1e-12:
        nx, ny, nz = ax / a_norm, ay / a_norm, az / a_norm
        cos_term = np.cos(a_norm * t)
        sin_term = np.sin(a_norm * t)

        n_dot_sigma = nx * sx + ny * sy + nz * sz
        for i in range(2):
            for j in range(2):
                val = cos_term * I2[i, j] - 1j * sin_term * n_dot_sigma[i, j]
                # Apply complex phase: phase = phase_real + i * phase_imag
                U[i, j] = phase_real * val - 1j * phase_imag * val
    else:
        for i in range(2):
            for j in range(2):
                U[i, j] = phase_real * I2[i, j] - 1j * phase_imag * I2[i, j]

    return U

@njit
def compute_rabi_signal(ne, ng, Rabi_freq, delta, T, f_peak, time_seq):
    sig = np.zeros(len(time_seq), dtype=np.complex128)

    for n_i in range(len(ne)):
        rho0 = np.array([[ne[n_i], 0.0], [0.0, ng[n_i]]], dtype=np.complex128)
        delta_i = delta[n_i] + (np.mean(T) - f_peak) * 1000 * 2 * np.pi
        H_mol = Eff_H(Rabi_freq[n_i], delta_i)

        for t_i in range(len(time_seq)):
            U = expm_hamiltonian_numba(H_mol, time_seq[t_i])
            Udag = U.conj().T
            rho_i = matmul_numba(matmul_numba(U, rho0), Udag)
            temp = matmul_numba(rho_i, sz)
            sig[t_i] += temp[0, 0] + temp[1, 1]  # Trace manually

    return sig/len(ne)

@njit
def Generate_molecule_set(Number_of_molecules):
    phi_01=np.random.uniform(0, 2*np.pi, Number_of_molecules) # The first Euler angle
    chi_01 = np.random.uniform(0, 2 * np.pi, Number_of_molecules) # The third Euler angle
    cos_theta_01 = np.random.uniform(-1, 1, Number_of_molecules)   # Cosine of polar angle
    theta_01 = np.arccos(cos_theta_01) # The second Euler angle
    return phi_01, theta_01, chi_01
    
@njit
def generate_regular_sphere_points(N, r=1.0):
    points = []
    a = 4 * np.pi * r**2 / N
    d = np.sqrt(a)
    M_theta = int(round(np.pi / d))
    d_theta = np.pi / M_theta
    d_phi = a / d_theta

    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            #points.append((x, y, z))
            points.append((theta, phi))

    return np.array(points)

@njit
def tile_1d(a, k):
    n = len(a)
    out = np.empty(n * k, dtype=a.dtype)
    for i in range(k):
        for j in range(n):
            out[i * n + j] = a[j]
    return out

@njit
def repeat_1d(a, k):
    n = len(a)
    out = np.empty(n * k, dtype=a.dtype)
    for i in range(n):
        for j in range(k):
            out[i * k + j] = a[i]
    return out

@njit
def Generate_molecule_set_equidistributed(Number_of_molecules:int, number_of_orientations:int):
    phi_01 = np.linspace(0, np.pi, number_of_orientations) # The first Euler angle
    point_set = generate_regular_sphere_points(Number_of_molecules)  # shape: (N_point_set, 2)
    #phi_grid, theta_grid, chi_grid = np.meshgrid(phi_01, theta_01, chi_01, indexing='ij')
    theta_01 = point_set[:, 0]
    chi_01 = point_set[:, 1]
    # Flatten to 1D arrays
    phi_01_all = repeat_1d(phi_01, Number_of_molecules)
    theta_01_all = tile_1d(theta_01, number_of_orientations)
    chi_01_all = tile_1d(chi_01, number_of_orientations)
    
    return phi_01_all, theta_01_all, chi_01_all
    
@njit
def matmul_numba(a:complex, b:complex):
    n, m = a.shape
    p = b.shape[1]
    result = np.zeros((n, p), dtype=np.complex128)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i, j] += a[i, k] * b[k, j]
    return result

@njit
def Rabi_frequency_v2_numba(v1:complex, v2:complex, phi:float, theta:float, chi:float):
    C1=matmul_numba(np.conj(v1).reshape((1, 3)),matmul_numba(T1pS(np.exp(1j*chi)*(-1j*np.sin(phi)-np.cos(theta)*np.cos(phi))/np.sqrt(2)),v2.reshape((3, 1))))
    C2=matmul_numba(np.conj(v1).reshape((1, 3)),matmul_numba(T10S(-np.sin(theta)*np.cos(phi)),v2.reshape((3, 1))))
    C3=matmul_numba(np.conj(v1).reshape((1, 3)), matmul_numba(T1mS(np.exp(-1j*chi)*(-1j*np.sin(phi)+np.cos(theta)*np.cos(phi))/np.sqrt(2)),v2.reshape((3, 1))))
    #print(C1)
    return C1[0,0]+C2[0,0]+C3[0,0]

@njit
def E_cubic_root(B_0, D_0, E_0, theta, chi):
    p=p_function(B_0, D_0, E_0)
    q = q_function(B_0, D_0, E_0, theta, chi)
    angle=ang(p,q)
    E_k = np.zeros(3)
    for k in range(3):
        E_k[k] = (2 * np.sqrt(-p / 3) * np.cos(angle+2*np.pi*(k+1)/3))
    return np.sort(E_k)


@njit
def eigenvector_from_eigenvalue(H, lam):
    """
    Given a 3x3 Hermitian matrix H and one of its eigenvalues lam,
    compute the corresponding eigenvector.
    """
    A = H - np.eye(3, dtype=np.complex128) * lam

    # Use cross product of two rows of A as an eigenvector estimate
    r1 = A[0]
    r2 = A[2]

    # The null space direction is orthogonal to both rows
    v = np.cross(r1, r2)

    # If zero vector (rows linearly dependent), try other pair
    if np.all(np.abs(v) < 1e-10):
        r2 = A[1]
        r3 = A[2]
        v = np.cross(r2, r3)
    if np.all(np.abs(v) < 1e-10):
        r1 = A[0]
        r3 = A[1]
        v = np.cross(r1, r3)

    # Normalize
    norm = np.sqrt(np.sum(np.abs(v)**2))
    if norm > 1e-12:
        return v / norm
    else:
        return np.zeros(3, dtype=np.complex128)
    
@njit
def eigenvector_set(H, E):
    M=np.zeros((3,3), dtype=np.complex128)
    for i in range(len(E)):
        M[i]=eigenvector_from_eigenvalue(H, E[i])
    return M.T

@njit(inline='always')
def abs2_dot(a, b):
    result = 0.0 + 0.0j
    for i in range(a.shape[0]):
        result += np.conj(a[i]) * b[i]
    return np.real(result * np.conj(result))


def matrix_exponential_diag(A):
    """
    Compute the matrix exponential e^A using the diagonalization method.
    
    This function assumes that the square matrix A is diagonalizable.
    
    Parameters:
        A (np.ndarray): A square matrix (n x n).
        
    Returns:
        expA (np.ndarray): The matrix exponential e^A.
    """
    # Compute eigenvalues and eigenvectors of A.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Compute the diagonal matrix of the exponentials of the eigenvalues.
    exp_lambda = np.diag(np.exp(eigenvalues))
    
    # Compute the matrix exponential using the formula e^A = V exp(Λ) V^{-1}
    expA = eigenvectors @ exp_lambda @ np.linalg.inv(eigenvectors)
    #expA = eigenvectors @ exp_lambda @ eigenvectors.conj().T
    return expA



@njit
def matrix_exponential_diag_numba(A):
    # Compute eigenvalues and eigenvectors of A.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Compute the diagonal matrix of the exponentials of the eigenvalues.
    exp_lambda = np.diag(np.exp(eigenvalues))
    
    # Compute the matrix exponential using the formula e^A = V exp(Λ) V^{-1}
    expA = matmul_numba(eigenvectors, matmul_numba(exp_lambda, np.linalg.inv(eigenvectors)))
    #expA = eigenvectors @ exp_lambda @ eigenvectors.conj().T
    return expA    
#Define all the time-evolution operators; 
#functions labelled with t are using analytical formula for the exponential of matrix
@njit
def U_0t(delta1, Dss, t):
    theta=delta1*t/2
    gamma=Dss*t/4
    return np.array([[np.exp(-1j*theta-1j*gamma), 0, 0, 0],
                     [0, np.exp(1j*theta+1j*gamma), 0, 0],
                     [0, 0, np.exp(-1j*theta+1j*gamma), 0],
                     [0, 0, 0, np.exp(1j*theta-1j*gamma)]])

@njit
def U_1t(Omega1, delta1, Dss, t): #time-evolution operator for EYFP MW transition is being addressed
    #Already checked with diagonalization method, this function is correct
    delta_p = delta1+Dss/2
    delta_m = delta1-Dss/2
    Omega_p=np.sqrt(Omega1**2+delta_p**2)
    Omega_m=np.sqrt(Omega1**2+delta_m**2)
    theta_p=Omega_p*t/2
    theta_m=Omega_m*t/2
    M11 = np.cos(theta_p) - 1j*delta_p/Omega_p*np.sin(theta_p)
    M12 = -1j*Omega1*np.sin(theta_p)/Omega_p
    M22 = np.cos(theta_p)+1j*delta_p/Omega_p*np.sin(theta_p)
    M33 = np.cos(theta_m)-1j*delta_m/Omega_m*np.sin(theta_m)
    M34 = -1j*Omega1*np.sin(theta_m)/Omega_m
    M44 = np.cos(theta_m)+1j*delta_m/Omega_m*np.sin(theta_m)
    U = np.array([[M11, M12, 0, 0],
                  [M12, M22, 0, 0],
                  [0, 0, M33, M34],
                  [0, 0, M34, M44]])
    #np.cos(Omega_p*t/2)*I-1j*(delta_p*sigmaz+Omega*sigmax)/Omega_p*np.sin(Omega_p*t/2)
    #U_m = np.cos(Omega_m*t/2)*I-1j*(delta_m*sigmaz+Omega*sigmax)/Omega_m*np.sin(Omega_m*t/2)
    return U

@njit
def U_1(Omega1: complex, delta1:float, Dss:complex, t:float): #time-evolution operator for target spin MW transition is being addressed
    #DO NOT USE scipy, the scipy function sucks
    H0=np.array([[(delta1+Dss/2)/2, Omega1/2, 0,  0],
                 [np.conj(Omega1)/2, (-delta1-Dss/2)/2, 0, 0],
                 [0, 0, (delta1-Dss/2)/2,  Omega1/2],
                 [0, 0, np.conj(Omega1)/2, (-delta1+Dss/2)/2]])
    #print(np.linalg.eigh(H0))
    #print(H0)
    #U=sp.linalg.expm(-1j*H0*t)
    U=matrix_exponential_diag_numba(-1j*H0*t)
    return U

@njit
def U_2_pi_pulse(Omega2, delta1, Dss): #time-evolution operator for pi pulse on target spin is being addressed
    Omega_p=np.sqrt(Omega2**2+Dss**2/4)
    C1=1j*Dss/(2*Omega_p)
    C2=1j*Omega2/Omega_p
    C3=np.exp(1j*delta1*np.pi/(2*Omega_p))
    C4=np.exp(-1j*delta1*np.pi/(2*Omega_p))
    U=np.array([[-C1*C4, 0, -C2*C4, 0],[0, C1*C3, 0, -C2*C3],[-C2*C4, 0, C1*C4, 0],[0, -C2*C3, 0, -C1*C3]])
    return U

@njit
def U_2(Omega2:float, delta1: complex, Dss: complex, t:float): #time-evolution operator for target spin MW transition is being addressed
    #This function using the custom matrix exponential method
    H0=np.array([[(delta1+Dss/2)/2, 0, Omega2/2, 0],
                 [0, (-delta1-Dss/2)/2, 0, Omega2/2],
                 [Omega2/2, 0, (delta1-Dss/2)/2, 0],
                 [0, Omega2/2, 0, (-delta1+Dss/2)/2]])
    #print(np.linalg.eigh(H0))

    #U=sp.linalg.expm(-1j*H0*t)
    U=matrix_exponential_diag_numba(-1j*H0*t)
    return U

@njit
def U_3(Omega1: complex, Omega2:float, delta1: complex, Dss: complex, t:float): #time-evolution operator for both transitions are being addressed
    H0=np.array([[(delta1+Dss/2)/2, Omega1/2, Omega2/2, 0],
                 [np.conj(Omega1)/2, (-delta1-Dss/2)/2, 0, Omega2/2],
                 [Omega2/2, 0, (delta1-Dss/2)/2, Omega1/2],
                 [0, Omega2/2, np.conj(Omega1)/2, (-delta1+Dss/2)/2]])
    #print(np.linalg.eigh(H0))
    #U=sp.linalg.expm(-1j*H0*t)
    U=matrix_exponential_diag_numba(-1j*H0*t)
    return U


@njit(parallel=True)
def Transitions_and_couplings_calculation_numba(B0:float, D0:float, E0:float, d:float, Omega1:float, phi_01:float, theta_01:float, chi_01:float, theta_12:float, phi_12:float):
    #B0: magnetic field, D0, E0: ZFS, d: distance, Omega1: \gamma_B B_1, the Rabi frequency factor, 
    #t_pi: pi pulse duration, phi_01, theta_01, chi_01: Euler angles between Lab frame and D-tensor frame, theta_12, phi_12: inter-spin vector orientation in D-tensor frame
    N_mol=len(theta_01)
    Delta1 = np.zeros(N_mol)*(0+0j)
    Delta2 = np.zeros(N_mol)*(0+0j)
    Delta3 = np.zeros(N_mol)*(0+0j)
    nx = np.zeros(N_mol)
    ny = np.zeros(N_mol)
    nz = np.zeros(N_mol)
    Rabi_freq_xz = np.zeros(N_mol)*(0+0j)
    Rabi_freq_yz = np.zeros(N_mol)*(0+0j)
    Txz = np.zeros(N_mol)*(0+0j)
    Tyz = np.zeros(N_mol)*(0+0j)
    Txy = np.zeros(N_mol)*(0+0j)   
    nx0=0.4
    ny0=0.4
    nz0=0.2
    V = 52.16/1000
    TS_list = [T1mS(1), T10S(1), T1pS(1)]
    qs = [-1, 0, 1]
    for i in prange(N_mol):
        M1=eigenvector_set(H0(D0, E0, 0, 0.0, theta_01[i], chi_01[i]), np.array([-2*D0/3, D0/3-E0, D0/3+E0]))
        H_0 = H0(D0, E0, B0, phi_01[i], theta_01[i], chi_01[i])
        E_0 = E_cubic_root(B0, D0, E0, theta_01[i], chi_01[i])
        
        M0 = eigenvector_set(H_0, E_0)
        
        Txz[i]=E_0[2]-E_0[0]
        Tyz[i]=E_0[1]-E_0[0]
        Txy[i]=E_0[2]-E_0[1]
        
        eigvec=M0.T
        
        Rabi_freq_xz[i]=Omega1 * Rabi_frequency_v2_numba(eigvec[2], eigvec[0], phi_01[i], theta_01[i], chi_01[i])
        Rabi_freq_yz[i]=Omega1 * Rabi_frequency_v2_numba(eigvec[1], eigvec[0], phi_01[i], theta_01[i], chi_01[i])
        
        nx[i] = nx0 * abs2_dot(M1.T[2], M0.T[2]) + \
                ny0 * abs2_dot(M1.T[1], M0.T[2]) + \
                nz0 * abs2_dot(M1.T[0], M0.T[2])

        ny[i] = nx0 * abs2_dot(M1.T[2], M0.T[1]) + \
                ny0 * abs2_dot(M1.T[1], M0.T[1]) + \
                nz0 * abs2_dot(M1.T[0], M0.T[1])

        nz[i] = nx0 * abs2_dot(M1.T[2], M0.T[0]) + \
                ny0 * abs2_dot(M1.T[1], M0.T[0]) + \
                nz0 * abs2_dot(M1.T[0], M0.T[0])


        for q_idx in range(3):
            q = qs[q_idx]
            TS = TS_list[q_idx]
            Mat_prod = C_q(q, theta_01[i], chi_01[i], theta_12, phi_12) * matmul_numba(np.conj(eigvec), matmul_numba(TS, eigvec.T))
            Delta1[i] += Mat_prod[0,0]
            Delta2[i] += Mat_prod[1,1]
            Delta3[i] += Mat_prod[2,2]
        
    Txz_ave = np.mean(Txz)
    Tyz_ave=np.mean(Tyz)
    Dss_zx=V/d**3*(Delta3-Delta1)*1000*2*np.pi # Dipole coupling strength Dss for Tz-Tx transition, unit: MHz
    Dss_zy=V/d**3*(Delta2-Delta1)*1000*2*np.pi # Dipole coupling strength Dss for Tz-Ty transition, unit: MHz
    deltaxz=(Txz-Txz_ave)*1000*2*np.pi #Tz-Tx transition detuning
    deltayz=(Tyz-Tyz_ave)*1000*2*np.pi #Tz-Ty transition detuning
    
    return Txz,Tyz, deltaxz, deltayz, Rabi_freq_xz, Rabi_freq_yz, Dss_zx, Dss_zy, nx, ny, nz


@njit(parallel=True)
def DEER_4_pulse_numba(Rabi_freq: complex, t_pi: float, Omega1: float, delta: float, Dss:complex, tau1: float, tau2: float, N_tau:int, n_g:float, n_e:float): #n_g, n_e: population on ground/excited states
    #Numba version of the 4-pulse DEER simulation
    tau_total=tau1+tau2
    tau=np.linspace(0, tau_total, N_tau)
    N_mol=len(Rabi_freq)
    sig_p_all = np.zeros((N_mol, N_tau))
    sig_m_all = np.zeros((N_mol, N_tau))
    #sig_c_all = np.zeros((N_mol, N_tau))
    #print('definition-0')
    sig_p=np.zeros_like(tau) #coupling to spin up
    sig_m=np.zeros_like(tau) #coupling to spin down
    sig_c=np.zeros_like(tau) #no coupling, for comparison only
    #rho_0p = np.zeros((4, 4), dtype=np.complex128)
    #rho_0m = np.zeros((4, 4), dtype=np.complex128)
    Dssi=0.0+0.0j
    

    O_s = np.array([[1/2,0, 0, 0], [0,-1/2, 0, 0],[0,0, 1/2, 0],[0,0, 0, -1/2]]) #Observation operator for EYFP spin
    #print('definition-1')
    for i in prange(N_mol):
        rho_0p = np.zeros((4, 4), dtype=np.complex128)
        rho_0m = np.zeros((4, 4), dtype=np.complex128)
        rho_0p[0, 0] = n_g[i]
        rho_0p[1, 1] = n_e[i]
        rho_0p[2, 2] = 0.0
        rho_0p[3, 3] = 0.0

        rho_0m[0, 0] = 0.0
        rho_0m[1, 1] = 0.0
        rho_0m[2, 2] = n_g[i]
        rho_0m[3, 3] = n_e[i]
        delta_1 = delta[i]
        Dssi = Dss[i]
        for j in range(N_tau):
        
            #print('Start of the (', i, ', ', j, ')th for-loops')
            
            #delta_1 = delta[i] #detuning delta_1 for eYFP MW pumping, unit: MHz
            #Dssi = Dss[i] #Unit: MHz
            #print('Definition of rhos in the (', i, ', ', j, ')th for-loops')
            #Pulse sequence
            U1=U_1t(np.abs(Rabi_freq[i]), delta_1, Dssi, t_pi/2) #pi/2 pulse on EYFP
            U_fe_1= U_0t(delta_1, Dssi, tau1) #'fe' means free evolution
            U2=matmul_numba(U1,U1) # pi pulse on EYFP
            U_fe_2= U_0t(delta_1, Dssi, tau[j])
            U3=U_2_pi_pulse(Omega1, delta_1, Dssi) #the third pi pulse, applied to the target spin
            U_fe_3= U_0t(delta_1, Dssi, tau_total-tau[j])
            U_fe_4= U_0t(delta_1, Dssi, tau2)
            U_DEER_4=matmul_numba(U1, matmul_numba(U_fe_4, matmul_numba(U2, matmul_numba(U_fe_3, matmul_numba(U3, matmul_numba(U_fe_2, matmul_numba(U2,matmul_numba(U_fe_1,U1)))))))) #4-pulse DEER sequence
            #print('Definition of Us in the (', i, ', ', j, ')th for-loops')        
            #U10=U_1t(Rabi_freq[i], delta_1, 0, t_pi/2)
            #U_fe_10= U_0t(delta_1, 0, tau1)
            #U20=matmul_numba(U10,U10) # pi pulse on EYFP
            #U_fe_20= U_0t(delta_1, 0, tau[j])
            #U30=U_2_pi_pulse(Omega1, delta_1, 0) #the third pi pulse, applied to the target spin
            #U_fe_30= U_0t(delta_1, 0, tau_total-tau[j])
            #U_fe_40= U_0t(delta_1, 0, tau2)
            #U_SE = matmul_numba(U10, matmul_numba(U_fe_40, matmul_numba(U20, matmul_numba(U_fe_30, matmul_numba(U30, matmul_numba(U_fe_20, matmul_numba(U20, matmul_numba(U_fe_10,U10))))))))
            #print(U_SE)
            
            #rho_tp =matmul_numba(U_DEER_4, matmul_numba(rho_0p, np.conj(U_DEER_4).T))
            #rho_tm =matmul_numba(U_DEER_4, matmul_numba(rho_0m, np.conj(U_DEER_4).T))
            #rho_t0 =matmul_numba(U_SE, matmul_numba(rho_0p, np.conj(U_SE).T))
            #sig_p[j]+=np.trace(matmul_numba(rho_tp, O_s)).real
            #sig_m[j]+=np.trace(matmul_numba(rho_tm, O_s)).real
            rho_tp = matmul_numba(U_DEER_4, matmul_numba(rho_0p, np.conj(U_DEER_4).T))
            rho_tm = matmul_numba(U_DEER_4, matmul_numba(rho_0m, np.conj(U_DEER_4).T))
            sig_p_all[i, j] = np.trace(matmul_numba(rho_tp, O_s)).real
            sig_m_all[i, j] = np.trace(matmul_numba(rho_tm, O_s)).real
            #sig_c[j]+=np.trace(matmul_numba(rho_t0, O_s)).real
    sig_p = np.sum(sig_p_all, axis=0) / N_mol
    sig_m = np.sum(sig_m_all, axis=0) / N_mol
    sig_c=sig_c/N_mol
    return tau, sig_p, sig_m, sig_c

@njit(parallel=True)
def DEER_4_pulse_numba_samples(Rabi_freq: complex, t_pi: float, Omega1: float, delta: float, Dss:complex, tau1: float, tau2: float, t_i: float, n_g:float, n_e:float): #n_g, n_e: population on ground/excited states
    #Numba version of the 4-pulse DEER simulation
    tau_total=tau1+tau2
    #tau=np.linspace(0, tau_total, N_tau)
    N_mol=len(Rabi_freq)
    N_t_i=len(t_i)
    sig_p_all = np.zeros((N_mol, N_t_i))
    sig_m_all = np.zeros((N_mol, N_t_i))

    sig_p=np.zeros(N_t_i) #coupling to spin up
    sig_m=np.zeros(N_t_i) #coupling to spin down
    Dssi=0.0+0.0j
    

    O_s = np.array([[1/2,0, 0, 0], [0,-1/2, 0, 0],[0,0, 1/2, 0],[0,0, 0, -1/2]]) #Observation operator for EYFP spin
    #print('definition-1')
    for i in prange(N_mol):
        
        rho_0p = np.zeros((4, 4), dtype=np.complex128)
        rho_0m = np.zeros((4, 4), dtype=np.complex128)
        rho_0p[0, 0] = n_g[i]
        rho_0p[1, 1] = n_e[i]
        rho_0p[2, 2] = 0.0
        rho_0p[3, 3] = 0.0

        rho_0m[0, 0] = 0.0
        rho_0m[1, 1] = 0.0
        rho_0m[2, 2] = n_g[i]
        rho_0m[3, 3] = n_e[i]
        delta_1 = delta[i]
        Dssi = Dss[i]
        for j in range(N_t_i):
            #Pulse sequence
            U1=U_1t(np.abs(Rabi_freq[i]), delta_1, Dssi, t_pi/2) #pi/2 pulse on EYFP
            U_fe_1= U_0t(delta_1, Dssi, tau1) #'fe' means free evolution
            U2=matmul_numba(U1,U1) # pi pulse on EYFP
            U_fe_2= U_0t(delta_1, Dssi, t_i[j])
            U3=U_2_pi_pulse(Omega1, delta_1, Dssi) #the third pi pulse, applied to the target spin
            U_fe_3= U_0t(delta_1, Dssi, tau_total-t_i[j])
            U_fe_4= U_0t(delta_1, Dssi, tau2)
            U_DEER_4=matmul_numba(U1, matmul_numba(U_fe_4, matmul_numba(U2, matmul_numba(U_fe_3, matmul_numba(U3, matmul_numba(U_fe_2, matmul_numba(U2,matmul_numba(U_fe_1,U1)))))))) #4-pulse DEER sequence

            rho_tp = matmul_numba(U_DEER_4, matmul_numba(rho_0p, np.conj(U_DEER_4).T))
            rho_tm = matmul_numba(U_DEER_4, matmul_numba(rho_0m, np.conj(U_DEER_4).T))
            sig_p_all[i, j] = np.trace(matmul_numba(rho_tp, O_s)).real
            sig_m_all[i, j] = np.trace(matmul_numba(rho_tm, O_s)).real
    sig_p = np.sum(sig_p_all, axis=0) / N_mol
    sig_m = np.sum(sig_m_all, axis=0) / N_mol
    #sig_c=sig_c/N_mol
    return sig_p, sig_m
    
@njit
def lorentzian_numba(f, f0, gamma):
    """Numba-compatible Lorentzian line shape (centered at f0)."""
    return (0.5 * gamma) / ((f - f0)**2 + (0.5 * gamma)**2) / np.pi

@njit
def convolve_spectrum_numba(f_lines, I_lines, f_axis, gamma):
    """Convolve transitions with Lorentzian lineshape using Numba."""
    spectrum = np.zeros_like(f_axis)
    for i in range(len(f_lines)):
        f0 = f_lines[i]
        I0 = I_lines[i]
        for j in range(len(f_axis)):
            spectrum[j] += I0 * lorentzian_numba(f_axis[j], f0, gamma)
    return spectrum