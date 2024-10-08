import numpy as np
from scipy.linalg import eig
from scipy.sparse import diags
from scipy.optimize import newton
import matplotlib.pyplot as plt
import matplotlib
from orr_sommerfeld_system import *
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 16}

# matplotlib.rc('font', **font)


def find_most_unstable_eigenvalues(Re_range, alpha_range, N, L, U, dU, d2U, file_prefix='data'):
    """
    Iterate over ranges of Re and alpha to find the most unstable eigenvalue for each combination.
    Returns matrices of Re, alpha, and the corresponding most unstable eigenvalue.
    """
    Re_mesh, alpha_mesh = np.meshgrid(Re_range, alpha_range, indexing='ij')
    most_unstable_eigenvalue = np.zeros_like(Re_mesh, dtype=np.complex128)

    for i, Re in enumerate(Re_range):
        print(i)
        for j, alpha in enumerate(alpha_range):
            eigenvalues, _, _ = solve_orr_sommerfeld_fd(Re, N, L, U, dU, d2U, alpha)
            # Find the most unstable (rightmost) eigenvalue
            most_unstable = eigenvalues[np.argmax(eigenvalues.real)].real
            most_unstable_eigenvalue[i, j] = most_unstable

    # Save the data to text files
    np.savetxt(f'{file_prefix}_Re_mesh.txt', Re_mesh, header='Reynolds number mesh')
    np.savetxt(f'{file_prefix}_alpha_mesh.txt', alpha_mesh, header='Wave number mesh')
    np.savetxt(f'{file_prefix}_most_unstable_eigenvalues.txt', most_unstable_eigenvalue.real, header='Most unstable eigenvalues (real part)')

    return Re_mesh, alpha_mesh, most_unstable_eigenvalue

def load_mesh_data(file_prefix='data'):
    """
    Load mesh and eigenvalue data from text files.
    Returns the Reynolds number mesh, wave number mesh, and most unstable eigenvalue matrix.
    """
    Re_mesh = np.loadtxt(f'{file_prefix}_Re_mesh.txt')
    alpha_mesh = np.loadtxt(f'{file_prefix}_alpha_mesh.txt')
    most_unstable_eigenvalues = np.loadtxt(f'{file_prefix}_most_unstable_eigenvalues.txt')

    return Re_mesh, alpha_mesh, most_unstable_eigenvalues

def newton_method_contour(initial_guess, N, L, U, dU, d2U):
    # Define range of alpha values
    # Define range of alpha values
    alpha_values = np.linspace(1.0, 1.04, 100)
    
    # Initialize arrays to store results
    alpha_results = []
    Re_results = []

    for i, alpha in enumerate(alpha_values):
        print(f'Iteration {i+1}/{len(alpha_values)}')
        # Use previous result as initial guess
        initial_guess = Re_results[-1] if Re_results else 5700
        
        # Find Re using Newton's method
        # Re = newton_method_single(initial_guess, N, L, U, dU, d2U, alpha)
        Re = newton(c_real_function, initial_guess, tol=1e-5, args=(N, L, U, dU, d2U, alpha))

        if Re is not None:
            alpha_results.append(alpha)
            Re_results.append(Re)

    # print(newton_method_single(5690, N, L, U, dU, d2U, 1.017))

    return alpha_results, Re_results

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

    # alpha_results, Re_results = newton_method_contour(5700, N, L, U, dU, d2U)
    # np.savez('arrays.npz', array1=alpha_results, array2=Re_results)

    # index = np.argmin(Re_results)
    # print(f"Re: {Re_results[index]}, alpha: {alpha_results[index]}")

    data = np.load('arrays.npz')

    # Access individual arrays by their keys
    alpha_results = data['array1']
    Re_results = data['array2']

    index = np.argmin(Re_results)
    print(f"Re: {Re_results[index]}, alpha: {alpha_results[index]}")

    plt.plot(Re_results, alpha_results)
    plt.xlabel('Re')
    plt.ylabel('alpha')
    plt.grid()
    plt.show()

    # print(newton(c_real_function, 5700, tol=1e-5, args=(N, L, U, dU, d2U, alpha_results[index])))