import numpy as np
from scipy.linalg import eig
from scipy.sparse import diags
from scipy.optimize import newton
import matplotlib.pyplot as plt
import matplotlib
from orr_sommerfeld_system import *
from orr_contour import *
from matplotlib.colors import Normalize
from matplotlib import cm

font = {'size'   : 16}

matplotlib.rc('font', **font)


def single_parameter_plotter(Re, N, L, U, dU, d2U, alpha):
    """Plot the eigenvalues of the Orr-Sommerfeld equation for a single Reynolds number and wave number."""
    eigenvalues, eigenfunctions, x = solve_orr_sommerfeld_fd(Re, N, L, U, dU, d2U, alpha)

    # Plot the eigenvalues
    plt.figure(figsize=(8, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', color='blue')
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.axis([-.9, .1, -1, 0])
    plt.axhline(y=0, color='k', linestyle='-')
    plt.axvline(x=0, color='k', linestyle='-')
    plt.grid()
    plt.show()

    return eigenvalues, eigenfunctions, x

def spatial_plotter(Re, N, L, U, dU, d2U, alpha):
    """Plot the eigenfunctions of the Orr-Sommerfeld equation for a single Reynolds number and wave number."""
    eigenvalues, eigenfunctions, x = solve_orr_sommerfeld_fd(5000, 1000, L, U, dU, d2U, 1.0)

    plt.plot(x[1:-1], np.abs(eigenfunctions[:, np.argmax(eigenvalues.real)]))
    plt.xlabel('x')
    plt.ylabel('Eigenfunction amplitude')
    plt.title('Eigenfunctions of Orr-Sommerfeld Equation')
    plt.grid()
    plt.show()

def phi_func(x_2, alpha, phi_vector):
    
    length = len(phi_vector)
    x2_index = (x_2+1)/2 * (length-1)

    return phi_vector[int(x2_index)]


def velocity_field(x1, x2, t, phi, phi_derivative, alpha, Re, N, L, U, dU, d2U):

    length = len(phi)
    x2_index = int((x2+1)/2 * (length-1))

    v_1 = (phi_derivative[x2_index] * np.exp(eigenvalues[unstable_index]*t)*np.exp(1j*alpha*x1) + (1- x2**2)).real
    v_2 = (-1j*alpha*phi[x2_index]* np.exp(eigenvalues[unstable_index]*t)*np.exp(1j*alpha*x1)).real

    return v_1, v_2

def calculate_pathline(initial_position, total_time, dt,  phi, phi_derivative, alpha, Re, N, L, U, dU, d2U):
    num_steps = int(total_time / dt)
    positions = [initial_position]

    x1, x2 = initial_position

    for i in range(num_steps):
        vx, vy = velocity_field(x1, x2, i * dt,  phi, phi_derivative, alpha, Re, N, L, U, dU, d2U)
        x1 += vx * dt
        x2 += vy * dt
        positions.append((x1, x2))

    return positions
    

if __name__ == '__main__':
   
    # Example usage:
    Re =  5900  # Reynolds number
    N = 500  # Number of points
    L = 2.0  # Domain length
    alpha = 1.0 # Wave number

    # Plane Poiseuille flow
    U = lambda y: 1 - y**2
    dU = lambda y: -2 * y
    d2U = -2

    eigenvalues, eigenfunctions, x2 = single_parameter_plotter(Re, N, L, U, dU, d2U, alpha)

    unstable_index = np.argmax(eigenvalues.real)

    # print(eigenvalues[unstable_index])

    # plt.plot(x[1:-1], eigenfunctions[:,unstable_index])
    # plt.xlabel('x_2')
    # plt.ylabel('Eigenfunction amplitude')
    # plt.title('Eigenfunctions of Orr-Sommerfeld Equation')
    # plt.grid()
    # plt.show()

    phi =  eigenfunctions[:,unstable_index]
    phi = phi[::8]
    x2 = x2[::8]

    # print(len(phi))
    # print(len(x2))
    phi_derivative = np.gradient(phi,x2)
    print(len(phi))
    print(len(phi_derivative))
    t=1
    x1 = np.linspace(0, 10, 100)



    v_1_y = phi_derivative * np.exp(eigenvalues[unstable_index]*t) + (1- x2**2)
    v_1_x = np.exp(1j*alpha*x1) 
    print(v_1_y.shape)

    v_2_y = -1j*alpha*phi* np.exp(eigenvalues[unstable_index]*t)
    v_2_x = np.exp(1j*alpha*x1)


    # plt.plot(x2[1:-1], phi)
    # plt.xlabel('x_2')
    # plt.ylabel('Eigenfunction phi')
    # plt.grid()
    # plt.show()
   

    # plt.plot(v_1_y.real, x2[1:-1], label='Real part of v_1')
    # plt.plot(v_1_y.imag, x2[1:-1], label='Imaginary part of v_1')
    # plt.xlabel('v_1')
    # plt.ylabel('x_2')
    # plt.grid()
    # plt.legend()
    # plt.show()

    # plt.plot(v_2_y.real, x2[1:-1], label='Real part of v_2')
    # plt.plot(v_2_y.imag, x2[1:-1], label='Imaginary part of v_2')
    # plt.xlabel('v_2')
    # plt.ylabel('x_2')
    # plt.grid()
    # plt.legend()
    # plt.show()


    v_1_full = np.outer(v_1_y, v_1_x)
    v_2_full = np.outer(v_2_y, v_2_x)

    x1_mesh, x2_mesh = np.meshgrid(x1, x2)

    print(x1_mesh.shape)
    print(x2_mesh.shape)
    print(v_1_full.shape)
    print(v_2_full.shape)

    plt.quiver(x1_mesh, x2_mesh, v_1_full, v_2_full, headwidth=3, headlength=3, width=0.0005)
    plt.show()

    
    # Define initial position and time parameters
    initial_position = (0, 0.5)  # Initial position (x1, x2)
    total_time = 1000  # Total time duration
    dt = 0.1  # Time step size

    # Calculate the pathline
    pathline = calculate_pathline(initial_position, total_time, dt,   phi, phi_derivative, alpha, Re, N, L, U, dU, d2U)

    # Extract x1 and x2 coordinates for plotting
    x1_values = [pos[0] for pos in pathline]
    x2_values = [pos[1] for pos in pathline]


    # Define colormap and normalize
    cmap = plt.get_cmap('plasma')
    normalize = Normalize(vmin=0, vmax=total_time)

    # Plot the pathline
    for i in range(len(pathline) - 1):
        color = cmap(normalize(i * dt))
        plt.plot([x1_values[i], x1_values[i + 1]], [x2_values[i], x2_values[i + 1]], color=color)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    plt.colorbar(sm, label='Time')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()





