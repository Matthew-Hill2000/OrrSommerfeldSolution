import numpy as np
from scipy.linalg import eig
from scipy.sparse import diags
from scipy.optimize import newton
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
# from orr_contour import *

font = {'size'   : 16}

matplotlib.rc('font', **font)

def finite_difference_matrices(N, L):
    """
    Generate second-order central finite difference matrices for first and second derivatives,
    and a fourth-order central finite difference matrix for the fourth derivative,
    using N points over a domain of length L.
    """
    h = L / (N - 1)  # Step size
    x = np.linspace(-L / 2, L / 2, N)  # Domain from -L/2 to L/2

    # Second-order central difference for the first, second and fourth derivative
    D1 = diags([-1 / (2 * h), 0, 1 / (2 * h)], [-1, 0, 1], shape=(N, N)).toarray()
    D2 = diags([1 / h**2, -2 / h**2, 1 / h**2], [-1, 0, 1], shape=(N, N)).toarray()
    D4 = diags([1 / h**4, -4 / h**4, 6 / h**4, -4 / h**4, 1 / h**4], [-2, -1, 0, 1, 2], shape=(N, N)).toarray()

    # Apply boundary conditions of phi(0)=0 for D1, D2, D4 directly
    D1[0, :] = 0; D1[-1, :] = 0; D2[0, :] = 0; D2[-1, :] = 0; D4[0, :] = 0; D4[-1, :] = 0
    D4[1][1] += 1/h**4; D4[-2][-2] += 1/h**4  # first order derivative boundary condition
    D4[1][1] += (6/11)/h**4; D4[1][2] += (8/11)/h**4; D4[1][3] -= (3/11)/h**4; D4[-2][-2] += (6/11)/h**4; D4[-2][-3] += (8/11)/h**4; D4[-2][-4] -= (3/11)/h**4  # second order derivative boundary condition
    return D1, D2, D4, x

def solve_orr_sommerfeld_fd(Re, N, L, U, dU, d2U, alpha):
    """Solve the Orr-Sommerfeld equation using finite differences. """
    D1, D2, D4, x = finite_difference_matrices(N, L)    
    
    # Orr-Sommerfeld operators using finite differences
    A = (D4 - 2 * alpha**2 * D2 + alpha**4 * np.eye(N))/Re - 1j*alpha*np.diag(U(x))@D2 + 1j*alpha**3*np.diag(U(x)) + 1j*d2U*alpha*np.eye(N)
    B = (D2 - alpha**2*np.eye(N))

    # Enforce no-slip boundary conditions for Orr-Sommerfeld equation by zeroing out the first and last rows
    A = A[1:-1, 1:-1]
    B = B[1:-1, 1:-1]
    
    # Solve eigenvalue problem
    eigenvalues, eigenfunctions = eig(A, B)
    
    # Sort the eigenvalues by their real parts in descending order to get the rightmost ones
    idx = eigenvalues.real.argsort()[::-1]  # This sorts them in descending order by real part
    rightmost_eigenvalues, rightmost_eigenfunctions = eigenvalues[idx][:50], eigenfunctions[:, idx][:, :50]  # Select the rightmost 50 eigenvalues and eigenfunctions
    for i in range(2):
        index_delete = np.argmax(rightmost_eigenvalues.imag)
        rightmost_eigenvalues, rightmost_eigenfunctions = np.delete(rightmost_eigenvalues, index_delete), np.delete(rightmost_eigenfunctions, index_delete, axis=1)
        
    return rightmost_eigenvalues, rightmost_eigenfunctions, x


def c_real_function(Re, N, L, U, dU, d2U, alpha):
    eigenvalues, _, _ = solve_orr_sommerfeld_fd(Re, N, L, U, dU, d2U, alpha)
    c_real = eigenvalues[np.argmax(eigenvalues.real)].real
    return c_real

def exponential_func(x, a, b):
    return a * np.exp(-b * x)

if __name__ == '__main__':

    # Example usage:
    Re =  5797  # Reynolds number
    N = 500  # Number of points
    L = 2.0  # Domain length
    alpha = 1.02056 # Wave number

    # Plane Poiseuille flow
    U = lambda y: 1 - y**2
    dU = lambda y: -2 * y
    d2U = -2


    # Re_results = []
    # N_values = []
    # for i in range(1, 39):
    #     print(f'Iteration {i+1}/59')
    #     N = 100 + 50*i
    #     Re_results.append(newton(c_real_function, 5700, tol=1e-3, args=(N, L, U, dU, d2U, 1.02056)))
    #     N_values.append(N)

    # np.savez('error_arrays.npz', array1=N_values, array2=Re_results)

    print(newton(c_real_function, 5700, tol=1e-8, args=(4000, L, U, dU, d2U, 1.02056)))

    data = np.load('error_arrays.npz')

    # Access individual arrays by their keys
    N_values = data['array1']
    Re_results = data['array2']
    
    
    plt.plot(N_values, Re_results)
    plt.xlabel('N')
    plt.ylabel('Re')
    plt.grid()
    plt.show()