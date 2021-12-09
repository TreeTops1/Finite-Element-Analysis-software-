from typing import Counter, ValuesView
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import matplotlib.tri as tri
np.set_printoptions(suppress=True)
import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


integration_points = 1
youngs_modulus = 0
v = 0
h = 0
elements =[]
boundary_conditions =[]
loads =[]
array_matrices_K =[]
array_elements=[]
matrix_D = (youngs_modulus/(1-v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5*(1-v)]])

#integrations points for complete integration
array_integration_points = ([[-1/math.sqrt(3), -1/math.sqrt(3)], [1/math.sqrt(3), -1/math.sqrt(3)],
[1/math.sqrt(3), 1/math.sqrt(3)], [-1/math.sqrt(3), 1/math.sqrt(3)]])

#creates the [K] matrix
def stiffness_matrix(array_nodes,s,t,h,fx):
    n1 = 1/4*((1-s)*(1-t))
    n2 = 1/4*((1+s)*(1-t))
    n3 = 1/4*((1+s)*(1+t))
    n4 = 1/4*((1-s)*(1+t))

    dN1_ds = -1/4*(1-t)
    dN2_ds = 1/4*(1-t)
    dN3_ds = 1/4*(1+t)
    dN4_ds = -1/4*(1+t)

    dN1_dt = -1/4*(1-s)
    dN2_dt = -1/4*(1+s)
    dN3_dt = 1/4*(1+s)
    dN4_dt = 1/4*(1-s)

    #Matrix [B] = [Derivative] [N]
    #        |  dN1_dx    0     dN2_dx    0     dN3_dx    0     dN4_dx    0     |
    #   [B]= |    0     dN1_dy    0     dN2_dy    0     dN3_dy    0     dN4_dy  |
    #        |  dN1_dy  dN1_dx  dN2_dy  dN2_dx  dN3_dy  dN3_dx  dN4_dy  dN4_dx  |

    # Since matrix [B] has it terms in relation (x,y) and the shape fuction is in terms (t,s)
    #  the jacobian [j] and its determinant are needed.
    #       | dx/ds  dy/ds |
    # [J] = | dx/dt  dy/dt |

    #array_nodes = np.array([[3,1],[5,2],[5,5],[2,3]])------------------------
    #dx_ds = X1*dN1_ds+X2*dN2_ds+X3*dN3_ds+X4*dN4_ds
    dx_ds = (array_nodes[0] * dN1_ds + array_nodes[2]* dN2_ds + array_nodes[4] * dN3_ds +
            array_nodes[6]* dN4_ds)
    #dx_dt = X1*dN1_dt+X2*dN2_dt+X3*dN3_dt+X4*dN4_dt
    dx_dt = (array_nodes[0]* dN1_dt + array_nodes[2] * dN2_dt + array_nodes[4]* dN3_dt +
            array_nodes[6]* dN4_dt)
    #dy_ds = Y1*dN1_ds+Y2*dN2_ds+Y3*dN3_ds+Y4*dN4_ds
    dy_ds = (array_nodes[1] * dN1_ds + array_nodes[3]* dN2_ds + array_nodes[5]* dN3_ds +
            array_nodes[7]* dN4_ds)
    #dy_dt = Y1*dN1_dt+Y2*dN2_dt+Y3*dN3_dt+Y4*dN4_dt
    dy_dt = (array_nodes[1] * dN1_dt + array_nodes[3]* dN2_dt + array_nodes[5]* dN3_dt +
            array_nodes[7]* dN4_dt)

    j = np.array([[dx_ds, dy_ds], [dx_dt, dy_dt]])

    j_det = np.linalg.det(j)
    j_inv = np. linalg. inv(j)

    #                                 | dN1_ds |
    # shape fuctions dN1_dx = |J_inv| | dN1_dt |
    dN1_dxdy = j_inv @ np.array([[dN1_ds], [dN1_dt]])
    dN2_dxdy = j_inv @ np.array([[dN2_ds], [dN2_dt]])
    dN3_dxdy = j_inv @ np.array([[dN3_ds], [dN3_dt]])
    dN4_dxdy = j_inv @ np.array([[dN4_ds], [dN4_dt]])
    # creating the different rows of the matrix [B]
    matrix_C = np.array([float(dN1_dxdy[0]), 0, float(
        dN2_dxdy[0]), 0, float(dN3_dxdy[0]), 0, float(dN4_dxdy[0]), 0])
    matrix_D = np.array([0, float(dN1_dxdy[1]), 0, float(
        dN2_dxdy[1]), 0, float(dN3_dxdy[1]), 0, float(dN4_dxdy[1])])
    matrix_E = np.array([float(dN1_dxdy[1]), float(dN1_dxdy[0]), float(dN2_dxdy[1]), float(
        dN2_dxdy[0]), float(dN3_dxdy[1]), float(dN3_dxdy[0]), float(dN4_dxdy[1]), float(dN4_dxdy[0])])
    # appending the rows to complete the B matrix
    matrix_B = np.concatenate([[matrix_C], [matrix_D], [matrix_E]], axis=0)
    
    matrix_B_traspose = np.transpose(matrix_B)
    
    #               | 1  v     0      |
    # [D] = E/1-v^2 | v  1     0      |
    #               | 0  0  1/2*(1-v) |

    matrix_D = (youngs_modulus/(1-v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5*(1-v)]])
    # for simple integration [K] = 2 * 2 * h [B (0,0)]T [D] [B (0,0)] |J(0,0|
    #matrix_intermediate = 1*1*h * matrix_B_traspose @ matrix_D @ matrix_B * j_det

    matrix_K = fx*fx*h * matrix_B_traspose @ matrix_D @ matrix_B * j_det
    return (matrix_B, matrix_K)

def sub_matrices(K_matrix,position):
    sub_matrix =np.zeros((2,2))
    sub_matrix[0,0]=K_matrix[(position[0]-1)*2,(position[1]-1)*2]
    sub_matrix[0,1]=K_matrix[(position[0]-1)*2,(position[1]-1)*2+1]
    sub_matrix[1,0]=K_matrix[(position[0]-1)*2+1,(position[1]-1)*2]
    sub_matrix[1,1]=K_matrix[(position[0]-1)*2+1,(position[1]-1)*2+1]
    return(sub_matrix)

# converts quad elements into tri elements
def quads_to_tris(quads):
    tris = [[None for j in range(3)] for i in range(2*len(quads))]
    for i in range(len(quads)):
        j = 2*i
        n0 = quads[i][0]
        n1 = quads[i][1]
        n2 = quads[i][2]
        n3 = quads[i][3]
        tris[j][0] = n0
        tris[j][1] = n1
        tris[j][2] = n2
        tris[j + 1][0] = n2
        tris[j + 1][1] = n3
        tris[j + 1][2] = n0
    return tris

# plots a finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements):
    for element in elements:
        x = [nodes_x[int(element[i])] for i in range(len(element))]
        y = [nodes_y[int(element[i])] for i in range(len(element))]
        plt.fill(x, y, edgecolor='black', fill=False)
    

def scaling_nodal_deformations(deformations,nodes_scaled,nodes_average_dimension):
    scale= np.average(abs(deformations))
    for counter,i in enumerate (deformations):
        if abs(i) < 0.00000001:
            i =0
        if i != 0:
            i=(i*((0.1*nodes_average_dimension)/scale))
        else:
            i =0
        if (counter % 2) == 0:
            nodes_scaled[int(counter/2)][0]=nodes_scaled[int(counter/2)][0] + i
        else:
            nodes_scaled[int(((counter-1)/2))][1]= nodes_scaled[int(((counter-1)/2))][1] +i
    return(nodes_scaled, scale)

# plots a finite element mesh
def plot_fem_mesh_undeformed(nodes_x, nodes_y, elements):
    for element in elements:
        x = [nodes_x[int(element[i])] for i in range(len(element))]
        y = [nodes_y[int(element[i])] for i in range(len(element))]
        plt.gca().set_aspect('equal')
        plt.fill(x, y, edgecolor='blue',ls="--", fill=False)
        

#calculates the stress in the node
def nodal_stress (array_nodes,s,t,displacements):

    dN1_ds = -1/4*(1-t)
    dN2_ds = 1/4*(1-t)
    dN3_ds = 1/4*(1+t)
    dN4_ds = -1/4*(1+t)

    dN1_dt = -1/4*(1-s)
    dN2_dt = -1/4*(1+s)
    dN3_dt = 1/4*(1+s)
    dN4_dt = 1/4*(1-s)

    dx_ds = (array_nodes[0][0] * dN1_ds + array_nodes[1][0]* dN2_ds + array_nodes[2][0] * dN3_ds +
            array_nodes[3][0]* dN4_ds)

    dx_dt = (array_nodes[0][0]* dN1_dt + array_nodes[1][0] * dN2_dt + array_nodes[2][0]* dN3_dt +
            array_nodes[3][0]* dN4_dt)
    
    dy_ds = (array_nodes[0][1] * dN1_ds + array_nodes[1][1]* dN2_ds + array_nodes[2][1]* dN3_ds +
            array_nodes[3][1]* dN4_ds)
    
    dy_dt = (array_nodes[0][1] * dN1_dt + array_nodes[1][1]* dN2_dt + array_nodes[2][1]* dN3_dt +
            array_nodes[3][1]* dN4_dt)

    j = np.array([[dx_ds, dy_ds], [dx_dt, dy_dt]])

    j_inv = np. linalg. inv(j)

    dN1_dxdy = j_inv @ np.array([[dN1_ds], [dN1_dt]])
    dN2_dxdy = j_inv @ np.array([[dN2_ds], [dN2_dt]])
    dN3_dxdy = j_inv @ np.array([[dN3_ds], [dN3_dt]])
    dN4_dxdy = j_inv @ np.array([[dN4_ds], [dN4_dt]])
    # creating the different rows of the matrix [B]
    matrix_C = np.array([float(dN1_dxdy[0]), 0, float(
        dN2_dxdy[0]), 0, float(dN3_dxdy[0]), 0, float(dN4_dxdy[0]), 0])
    matrix_F = np.array([0, float(dN1_dxdy[1]), 0, float(
        dN2_dxdy[1]), 0, float(dN3_dxdy[1]), 0, float(dN4_dxdy[1])])
    matrix_E = np.array([float(dN1_dxdy[1]), float(dN1_dxdy[0]), float(dN2_dxdy[1]), float(
        dN2_dxdy[0]), float(dN3_dxdy[1]), float(dN3_dxdy[0]), float(dN4_dxdy[1]), float(dN4_dxdy[0])])
    # appending the rows to complete the matrix
    matrix_B = np.concatenate([[matrix_C], [matrix_F], [matrix_E]], axis=0)
    #               | 1  v     0      |
    # [D] = E/1-v^2 | v  1     0      |
    #               | 0  0  1/2*(1-v) |

    matrix_D = (youngs_modulus/(1-v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5*(1-v)]])
    # to obtain the stress at the point (t,s) = [D]  [B] {displacements}
    
    stress =  matrix_D @ matrix_B @ displacements
    return (stress)

def Plotting_von_mises(nodes_x,nodes_y,array_nodes_position,triangulation,von_mises_stress):
    plot_fem_mesh(nodes_x, nodes_y, array_nodes_position)
    plt.tricontourf(triangulation, von_mises_stress)
    # show
    plt.clim(np.amin(von_mises_stress),np.amax(von_mises_stress))
    plt.colorbar()
    plt.axis('equal')
    #plt.show() #used to plot outside the GUI window

def Plotting_stresses(nodes_x, nodes_y, array_nodes_position,triangulation, array_nodal_stresses):
    # plot the finite element mesh --- 
    plot_fem_mesh(nodes_x, nodes_y, array_nodes_position)
    #custom colourmap
    top = cm.get_cmap('Reds_r', 128)
    bottom = cm.get_cmap('viridis_r', 128) 
    newcolors = np.vstack((top(np.linspace(0.3, 0.5, 30)),
                       bottom(np.linspace(0, 1, 128))))
    cmap = ListedColormap(newcolors, name='OrangeBlue')
    
    #custom colourmap x 2
    #top = cm.get_cmap('gist_rainbow', 128)
    #cmap = ListedColormap(top(np.linspace(0, 0.75, 128)), name='OrangeBlue')
    plt.tricontourf(triangulation, array_nodal_stresses, cmap=cmap)

    #print(np.min(array_nodal_stresses))
    #print(np.max(array_nodal_stresses))
    
    # plot the contours
    
    #plt.tricontourf(triangulation, array_nodal_stresses)
    # show
    plt.colorbar()
    plt.clim(np.amin(array_nodal_stresses),np.amax(array_nodal_stresses))
    plt.axis('equal')
    #plt.show() #used to plot outside the GUI window

def Plotting_deformations(nodes_x, nodes_y, array_nodes_position,array_displacements_x,array_displacements_y,triangulation_deformed,deformations):
    total_deformation= []
    for i in range(0,len(deformations),2):
        n= deformations[i]
        j=deformations[i+1]
        total_deformation.append(math.sqrt(j**2+n**2))
    plot_fem_mesh_undeformed(nodes_x, nodes_y, array_nodes_position)
    plot_fem_mesh(array_displacements_x, array_displacements_y, array_nodes_position)       
    plt.tricontourf(triangulation_deformed, total_deformation)
    plt.clim(np.amin(total_deformation),np.amax(total_deformation))
    plt.colorbar()
    plt.axis('equal')
    #plt.show()  #used to plot outside the GUI window

def Printing_results (number_elements,number_nodes,direction,matrix_global,local_K_matrix,deformations,nodal_reactions=False,von_mises_stress=False,nodal_stresses=False):
    # writing the results in a txt.file
    with open(direction,"w") as f:
        
        f.write('············Finite Element Analysis ·············  \n\n'
                'Title:  \n'
                'Author:  \n\n'
                '\tAnalysis Information:  \n'
                'Number of Elements '+ str(number_elements) +'\n'
                'Number of Nodes '+ str(number_nodes) +'\n'
                '·················································\n\n'
                )

        #Printing K Global Matrix
        if matrix_global.any() != 0:
            f.write('\n\n\n············Matrix K Global············· \n')
            np.savetxt(f,np.round(matrix_global,3), fmt='%15.2f')
        
        #Printing K Matrix Elements
        if local_K_matrix == True:
            for a in array_elements:
                f.write('\n\n\n ············Matrix K Elements············\n')
                np.savetxt(f,a.K_matrix, fmt='%15.2f')       
        
        #Printing Nodal Displacements
        if deformations.any() != 0:
            f.write('\n\n\n ············Nodal displacements············ ')    
            for counter, i in enumerate(deformations):
                if (counter % 2) == 0:
                    f.write('\n'+'\n Node nº '+  str(counter/2+1) +'\n')
                if (counter % 2) == 0:  
                    f.write("\t(x) "+"{:10.10f}".format(i)+"  \t ")
                else: 
                    f.write("\t(y) "+"{:10.10f}".format(i)+" ")

        #Printing Element Displacements
        if deformations.any() != 0:
            f.write('\n\n\n············Element displacements············\n')    
            for counter, a in enumerate(array_elements):
                f.write('\n ·············Element nº '+  str(counter) + '············ \n')
                np.savetxt(f,a.displacements,fmt='%10.10f')        
                f.write('\n')
        
        #printing nodal stresses
        if nodal_stresses.any() != 0:
            for a in array_elements:
                f.write('\n\n\n············Nodal stresses············\n')    
                f.write('············Node nº ' + str(a.node_number)+ ' stresses, x, y, xy············\n')
                np.savetxt(f,a.stresses, fmt='%10.10f')  
        
        #Printing Nodal Stresses
        if nodal_stresses.any() != 0:
            f.write('\n\n\n············Nodal stresses, x, y, xy············ \n')      
            counter=0
            for  i,n in zip(nodal_stresses,von_mises_stress):
                counter= counter+1
                f.write(" Node "+str(counter)+"\n ") 
                f.write("\t      (x)  stress "+str(i[0])+"\n ") 
                f.write("\t      (y)  stress "+str(i[1])+"\n ") 
                f.write("\t      (xy) stress "+str(i[2])+"\n ") 
                f.write("\tEquivalent stress "+str(n)+"\n ") 


        #Printing Nodal reactions
        if nodal_reactions.any() != 0:
            f.write('\n\n\n············Nodal reactions············· ')
            for counter, i in enumerate(nodal_reactions):
                if (counter % 2) == 0:
                    f.write('\n'+'\n Node nº '+  str(counter/2+1) +'\n')
                if  (counter % 2) == 0:  
                    f.write("\t(x) force "+"{:10.10f}".format(i)+" ")   
                else: 
                    f.write("\t(y) force  "+"{:10.10f}".format(i)+" ") 

#creates a class for each element to storage [K], nodes, 
class Elements():
    def __init__(self, nodes, K_matrix,matrix_B,number):
        self.node_number=number
        self.nodes = nodes
        self.K_matrix = K_matrix
        self.node1 = [self.nodes[0],self.nodes[1]]
        self.node2 = [self.nodes[2],self.nodes[3]]
        self.node3 = [self.nodes[4],self.nodes[5]]
        self.node4 = [self.nodes[6],self.nodes[7]]
        self.nodes = np.array([[nodes[0],nodes[1]],[nodes[2],nodes[3]],[nodes[4],nodes[5]],[nodes[6],nodes[7]]])
        self.matrix_B=matrix_B
        self.matrix_D=(youngs_modulus/(1-v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5*(1-v)]])
        self.nodal_displacement =np.zeros((4,2))
        self.matrix_D = (youngs_modulus/(1-v**2)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5*(1-v)]])
        self.nodes_local=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        #self.nodes_local=([[-1/math.sqrt(3), -1/math.sqrt(3)], [1/math.sqrt(3), -1/math.sqrt(3)],[1/math.sqrt(3), 1/math.sqrt(3)], [-1/math.sqrt(3), 1/math.sqrt(3)]])

    #extracts the submatrix [k] of a node 
    def sub_matrices(self,position1,position2):
        sub_matrix =np.zeros((2,2))
        sub_matrix[0,0]=self.K_matrix[(position1-1)*2,(position2-1)*2]
        sub_matrix[0,1]=self.K_matrix[(position1-1)*2,(position2-1)*2+1]
        sub_matrix[1,0]=self.K_matrix[(position1-1)*2+1,(position2-1)*2]
        sub_matrix[1,1]=self.K_matrix[(position1-1)*2+1,(position2-1)*2+1]
        return(sub_matrix)

    #stores the nodal stress vector  
    def stress(self,displacements):
        self.displacements= displacements
        self.stresses=np.zeros((4,3))
        for counter, a in enumerate(self.nodes_local):           
            s=a[0]
            t=a[1]
            self.nodal_stress= nodal_stress(self.nodes,s,t,displacements)
            self.stresses[counter]=self.nodal_stress

    def nodal_displacements(self, displacements):
        self.nodal_displacement= displacements 

    def print_data (self):
        #print("nodes coordinates; ", self.nodes)
        #print("[K] = ; ", self.K_matrix)
        #print("B Matrix =", matrix_B)
        #print("desplazamientos = ", self.displacements)
        print("stress = ; ", self.stresses)

##############################################################################################################
           #  "C:\Users\Alfredo\Desktop\UNED\Proyecto master\Python pruebas\FEM_input.txt"
           #   C:/Users/Alfredo/Desktop/UNED/Proyecto master/Python pruebas/Datos_input.txt                                                                                    
##############################################################################################################
def Extracting_data(direction):
    #Extrating data from FEM_input.txt
    with open(direction) as f:
        contents = f.readlines()

    #removing empty spaces
    y1=re.split('\s+',contents[0])
    y1=re.split('\s+',contents[1])
    y2=re.split('\s+',contents[2])
    y3=re.split('\s+',contents[3])
    y4=re.split('\s+',contents[4])
    #asigning variables
    integration_points = float(y1[0])
    youngs_modulus = float(y2[0])
    #v = poisson value
    v = float(y3[0])
    #h = thickness
    h = float(y4[0])
    #calculating number of elements, boundary conditions & applied loads 
    number_elements =0
    number_Bc=0
    number_loads = 0
    for i in contents:
        n=re.split('\s+|,',i)
        if n[0] == "E:": 
            number_elements= number_elements+1
        if n[0] == "Bc:": 
            number_Bc= number_Bc+1
        if n[0] == "L:": 
            number_loads = number_loads+1
    #creating arrays with all the nodes, boundary conditons & applied loads
    elements =np.zeros(shape=(number_elements,8))#empty array to be filled with all the coordinates of the nodes
    boundary_conditions =np.zeros(shape=(number_Bc,2))#empty array to be filled with coordinates of all the fixed nodes
    loads =np.zeros(shape=(number_loads,4))#empty array to be filled with cooridnates of all the loads applied 
    contadorE =0 
    contadorBc=0 
    contadorL =0
    #filling the empty arrays with data
    for i in contents:
        n=re.split('\s+|,',i)
        if n[0] == "E:":
            elements[contadorE]=[n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8]]
            contadorE += 1
        if n[0] == "Bc:":
            boundary_conditions[contadorBc]=[n[1],n[2]]
            contadorBc+= 1
        if n[0] == "L:":
            loads[contadorL]=[n[1],n[2],n[3],n[5]]
            contadorL+= 1
    return(integration_points,youngs_modulus,v,h,number_elements,number_Bc,number_loads,elements,boundary_conditions,loads)
#the above function returns(integration_points,youngs_modulus,v,h,number_elements,number_Bc,number_loads,elements,boundary_conditions,loads)

def calculate(integration_points,youngs_modulus,v,h,number_elements,number_Bc,number_loads,elements,boundary_conditions,loads):
    #creating the [K] matrices for all elements
    counter=1
    for n in range (len(elements)):
        matrix_K = np.zeros((8, 8)) # creates a [0] [k] matrix for each element
        #Creating the K matrix for 1 or 4 integration points 
        for i in range(int(integration_points)):
            # fx=2 for simple integration or fx=1 for complete integration
            fx = 2
            if integration_points == 4:
                s = array_integration_points[i][0]
                t = array_integration_points[i][1]
                fx = 1
            elif integration_points == 1:
                s = 0
                t = 0
            else:
                print("error; integration points != 1 or 4")
                break
            matrix_B,b = stiffness_matrix(elements[n],s,t,h,fx)
            matrix_K = matrix_K + b #creating the [K] matrix       
        array_elements.append(Elements(elements[n],matrix_K,matrix_B,counter))#array with all the  elements (class)
        counter=counter+1
        array_matrices_K.append(matrix_K) #array with all the [K] matrices
   
    # creates an array with all the nodes coordinates 
    nodes_coordinates =[] 
    for n in range (len(elements)):
        a=array_elements[n]
        nodes_coordinates.append(a.node1)
        nodes_coordinates.append(a.node2)
        nodes_coordinates.append(a.node3)
        nodes_coordinates.append(a.node4)

    # creates an array with all the cooridnates of the nodes without duplicates 
    nodes_coor = np.array([x for n, x in enumerate(nodes_coordinates) if x not in nodes_coordinates[:n]])
    number_nodes=len(nodes_coor)
    
    #creating an array for plotting the elements.
    array_nodes_position=np.zeros((len(elements),4))
    for counter1, a in enumerate(elements):
        for j in range (int(len(a)/2)):
            for counter, n in enumerate(nodes_coor):
                if n[0]==a[int(j*2)] and n[1]==a[int(j*2+1)]:
                    array_nodes_position[counter1][j]=counter
    #saving array coordinates
    nodes_x =[]
    nodes_y=[]
    for i in nodes_coor:
        nodes_x.append(i[0])
        nodes_y.append(i[1])

    plot_fem_mesh_undeformed(nodes_x, nodes_y, array_nodes_position)

    # creates an empty K global matrix
    K_zeros=np.zeros((2,2))
    matrix_K_global=[]
    for i in range (len(nodes_coor)**2):
        matrix_K_global.append (K_zeros)

    # Creating an intermediate array with all the blocks K[2,2] placed in order
    for counter_i, i in enumerate (nodes_coor):
        for counter_n, n in enumerate(nodes_coor):
            for a in array_elements: #This evaluates all the elements
                for counter,j in enumerate(a.nodes): #This evaluates all nodes from all the elements
                    if j[0] == i[0] and j[1] == i[1]:
                        for counter_2,h in enumerate(a.nodes): #This evaluates all nodes from all the elements
                            if h[0] == n[0] and h[1] == n[1]:
                                index_matrix=int(counter_i*(len(nodes_coor))+counter_n)
                                matrix_K_global[index_matrix]=matrix_K_global[index_matrix]+a.sub_matrices(counter+1,counter_2+1)
    matrix_global= np.zeros((len(nodes_coor)*2,len(nodes_coor)*2))
    #creating a null global matrix [0]
    for i in range(0,len(nodes_coor)*2,2):
        for n in range(0,len(nodes_coor)*2,2):
            #replacing the K[2,2] submatrices of the global matrix for the submatrices of the matrix_K_global
            matrix_global[i:i+2,n:n+2]=matrix_K_global[int(i/2)*len(nodes_coor)+int(n/2)]
    #removing the fixed nodes in [K] global matrix
    nodes_position_removed=[]
    for i in boundary_conditions:
        for counter, n in enumerate (nodes_coor):
            if i[0] == n[0] and i[1]==n[1]:
                nodes_position_removed= np.append(nodes_position_removed,counter)
    nodes_position_removed=np.sort(nodes_position_removed)
    nodes_position_removed=nodes_position_removed[::-1]
    k_global_short=matrix_global.copy()
    for i in nodes_position_removed:
        k_global_short=np.delete(k_global_short,int(i)*2+1,1)
        k_global_short=np.delete(k_global_short,int(i)*2+1,0)
        k_global_short=np.delete(k_global_short,int(i)*2,1)
        k_global_short=np.delete(k_global_short,int(i)*2,0)
    
    k_global_short_inv = np.linalg.inv(k_global_short) 
    
    #finding the position of the loads in the array
    position_loads=[]
    for i in loads:
        for counter, n in enumerate (nodes_coor):
            if i[0] == n[0] and i[1]==n[1]:

                position_loads= np.append(position_loads,counter)
    array_loads=np.zeros((len(nodes_coor)*2))

    #creating an array with the loads
    for i,n in zip(position_loads, loads):
        array_loads[int(i*2)]=n[2]
        array_loads[int(i*2+1)]=n[3]

    #removing the foces that matches with the fixed nodes
    for i in nodes_position_removed:
        array_loads=np.delete(array_loads,int(i)*2+1)
        array_loads=np.delete(array_loads,int(i)*2)
    
    #### obtaning the deformation vector by multiplying the reduced K matrix and the loads array
    deformations=k_global_short_inv @ array_loads
    # rearranging the positions removed in the reduced K matrix
    nodes_position_removed=np.sort(nodes_position_removed)
    for i in nodes_position_removed:
        deformations = np.insert(deformations,int(i*2),0)
        deformations = np.insert(deformations,int(i*2),0)
    
    #### obtaning the forces vector bu multiplying the matrix global and the deformations
    forces = matrix_global @ deformations

    #creating an automatic scale for ploting deformations 
    #first an average node size is required 
    array_leng_diagonal=[]
    for i in array_elements:
        nodes_diagonal=i.nodes
        leng_diagonal=math.sqrt(abs(((nodes_diagonal[2][1]-nodes_diagonal[0][1])**2+(nodes_diagonal[2][0]-nodes_diagonal[0][0])**2)))
        array_leng_diagonal=np.append(array_leng_diagonal,leng_diagonal)
    nodes_average_dimension=(np.average(array_leng_diagonal) /math.sqrt(2)) 
    nodes_scaled = nodes_coor.copy()
    nodes_scaled, scale=scaling_nodal_deformations(deformations,nodes_scaled,nodes_average_dimension)

    #saving array displacements
    array_displacements_x =[]
    array_displacements_y=[]
    for i in nodes_scaled:
        array_displacements_x.append(i[0])
        array_displacements_y.append(i[1])

    #this section saves the nodal displacements inside the element class
    for i in array_elements:
        displacemetents =np.zeros(8)
        for counter2, n in enumerate(i.nodes):
            for counter, j in enumerate(nodes_coor):
                comparison = j == n
                if comparison.all() ==True:
                    displacemetents[counter2*2]=deformations[counter*2]
                    displacemetents[counter2*2+1]=deformations[counter*2+1]
        i.stress(displacemetents)
        
    array_nodal_stresses=np.zeros((len(nodes_coor),3))
    # averaging nodal stress 
    for counter_i, i in enumerate (nodes_coor):
        counter2=0
        for a in array_elements: #it evaluates all the elements    
            for counter,j in enumerate(a.nodes): #it evaluates all nodes from all the elements
                comparison = i == j
                if comparison.all() == True: # if two nodes from diferent elements matches then it averages the forces 
                    counter2=counter2+1
                    if counter2 < 3:
                        array_nodal_stresses[counter_i]=(array_nodal_stresses[counter_i]+a.stresses[counter])/counter2
                    if counter2 == 3:
                        array_nodal_stresses[counter_i]=array_nodal_stresses[counter_i]*2
                        array_nodal_stresses[counter_i]=(array_nodal_stresses[counter_i]+a.stresses[counter])/counter2
                    if counter2 == 4:
                        array_nodal_stresses[counter_i]=array_nodal_stresses[counter_i]*3
                        array_nodal_stresses[counter_i]=(array_nodal_stresses[counter_i]+a.stresses[counter])/counter2    
           

    array_nodal_stresses_x=np.zeros((len(nodes_coor)))
    array_nodal_stresses_y=np.zeros((len(nodes_coor)))
    array_nodal_stresses_xy=np.zeros((len(nodes_coor)))
    for counter, i in enumerate (array_nodal_stresses):
        array_nodal_stresses_x[counter]=i[0]
        array_nodal_stresses_y[counter]=i[1]
        array_nodal_stresses_xy[counter]=i[2]

    # convert all elements into triangles
    elements_all_tris = quads_to_tris(array_nodes_position)

    # create an unstructured triangular grid instance
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)
    triangulation_deformed= tri.Triangulation(array_displacements_x, array_displacements_y, elements_all_tris)
    #calculating vin mises stress for all nodes #V = √(σx^2 - (σx * σy) + σy^2 + (3 *t^2))
    von_mises_stress=np.zeros((len(array_nodal_stresses)))
    for counter, i in enumerate(array_nodal_stresses):
        von_mises_stress[counter]=math.sqrt(i[0]**2-(i[0]*i[1])+i[1]**2+(3*i[2]**2))
    return(nodes_x,nodes_y,array_nodes_position,array_displacements_x,array_displacements_y,
            triangulation,array_nodal_stresses_xy,array_nodal_stresses_x,array_nodal_stresses_y,von_mises_stress,scale,
            matrix_global,forces,deformations,array_nodal_stresses,number_nodes,triangulation_deformed)

#nodes_x=calculating[0]
#nodes_y=calculating[1]
#array_nodes_position=calculating[2]
#array_displacements_x= calculating[3]
#array_displacements_y=calculating[4]
#triangulation=calculating[5]
#array_nodal_stresses_xy=calculating[6]
#array_nodal_stresses_x=calculating[7]
#array_nodal_stresses_y=calculating[8]
#von_mises_stress=calculating[9]
#scale=calculating[10]
#matrix_global=calculating[11]
#forces=calculating[12]
#deformations=calculating[13]
#array_nodal_stresses=calculating[14]
#number_nodes=calculating[15]
#triangulation_deformed=calculating[16]
##################################################################################################
#
#                                      Creating the GUI
#
###################################################################################################


# New figure and plot variables so we can manipulate them
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False}

# Helper Functions
# Dibuja la figura
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Any 16'
sg.theme('SandyBeach')

#crea la columna de la izquierda
file_list_column = [    [sg.Text("Select path of the input file .txt")],
                        [sg.Input(), sg.FileBrowse(key="-IN-")],
                        [sg.Button('Calculate', font=(AppFont,10)), sg.Text("It may take a few seconds")],
                        [sg.Text(" " * 50)],
                        [sg.Text("---------------------Postprocessing---------------------", size=(45, 1), justification="c")],
                        [sg.Button('Deformation', font=(AppFont,10)), sg.Listbox(values=(''),
                                        size=(10, 1), key='_LISTBOX_', no_scrollbar=True),sg.Text("Scale")],
                        [sg.Button('Stress X', font=(AppFont,10)),sg.Button('Stress Y', font=(AppFont,10)),
                         sg.Button('Stress XY', font=(AppFont,10)),sg.Button('Von misses Stress', font=(AppFont,10))],
                        [sg.Text(" " * 50)],
                        [sg.Text("---------------------Priting results---------------------", size=(45,1), justification="c")],
                        [sg.Checkbox("Global K Matrix",key="K_matrix")],
                        [sg.Checkbox("Element´s K Matrix",key="Local_K_matrix")],
                        [sg.Checkbox("Nodal Reactions", key="reactions")],
                        [sg.Checkbox("Nodal Stresses",key="nodal_stresses")],
                        [sg.Checkbox("Nodal Deformations",key="Deformations")],
                        [sg.Text("Select path of the output file .txt")],
                        [sg.Input(), sg.FileBrowse(key="-OUT-")],
                        [sg.Button('Print results', font=(AppFont,10))],
                        [sg.Button('Exit', font=(AppFont,10))]   ]

layout = [  [sg.Column(file_list_column), sg.VSeperator(),sg.Canvas(key='figCanvas')]  ]
_VARS['window'] = sg.Window('Finite Element Analysis',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            icon =(r"C:\Users\Alfredo\Desktop\UNED\Proyecto master\icono2.ico"),
                            element_justification="right")

# \\  -------- PYSIMPLEGUI -------- //

# \\  -------- PYPLOT -------- //

#esto dibuja el grafico de PLYPOT
def drawChart():
    _VARS['pltFig'] = plt.figure()
    plot_fem_mesh_undeformed(nodes_x, nodes_y, array_nodes_position)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateChart_deformation():
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()#esto cierra la ventana
    _VARS['pltFig'] = plt.figure()
    fig =Plotting_deformations(nodes_x, nodes_y, array_nodes_position,array_displacements_x,array_displacements_y,triangulation_deformed,deformations)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateChart_stressX():
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()#esto cierra la ventana
    _VARS['pltFig'] = plt.figure()
    fig =Plotting_stresses(nodes_x, nodes_y, array_nodes_position,triangulation, array_nodal_stresses_x)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateChart_stressY():
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()#esto cierra la ventana
    _VARS['pltFig'] = plt.figure()
    fig =Plotting_stresses(nodes_x, nodes_y, array_nodes_position,triangulation, array_nodal_stresses_y)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateChart_stressXY():
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()#esto cierra la ventana
    _VARS['pltFig'] = plt.figure()
    fig =Plotting_stresses(nodes_x, nodes_y, array_nodes_position,triangulation, array_nodal_stresses_xy)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def updateChart_Von_misses_Stress():
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.clf()#esto cierra la ventana
    _VARS['pltFig'] = plt.figure()
    fig =Plotting_von_mises(nodes_x,nodes_y,array_nodes_position,triangulation,von_mises_stress)
    _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)  #esto crea la ventana
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Calculate':
        data_FEM=Extracting_data(values["-IN-"])
        integration_points=data_FEM[0]
        youngs_modulus=data_FEM[1]
        v=data_FEM[2]
        h=data_FEM[3]
        number_elements=data_FEM[4]
        number_Bc=data_FEM[5]
        number_loads=data_FEM[6]
        elements=data_FEM[7]
        boundary_conditions=data_FEM[8]
        loads=data_FEM[9]

        calculating=calculate(integration_points,youngs_modulus, v, h, 
        number_elements, number_Bc, number_loads, elements, boundary_conditions, loads)
        nodes_x=calculating[0]
        nodes_y=calculating[1]
        array_nodes_position=calculating[2]
        array_displacements_x= calculating[3]
        array_displacements_y=calculating[4]
        triangulation=calculating[5]
        array_nodal_stresses_xy=calculating[6]
        array_nodal_stresses_x=calculating[7]
        array_nodal_stresses_y=calculating[8]
        von_mises_stress=calculating[9]
        matrix_global=calculating[11]
        nodal_reactions=calculating[12]
        deformations=calculating[13]
        nodal_stresses=calculating[14]
        number_nodes=calculating[15]
        triangulation_deformed=calculating[16]
        drawChart()
        scale = np.round(0.1/calculating[10],2)
        _VARS['window'].Element('_LISTBOX_').Update(values=[scale])

    if event == 'Deformation':
        updateChart_deformation()
    if event == 'Stress X':
        updateChart_stressX()
    if event == 'Stress Y':
        updateChart_stressY()
    if event == 'Stress XY':
        updateChart_stressXY()
    if event == 'Von misses Stress':
        updateChart_Von_misses_Stress()
    if event == 'Print results':
        local_K_matrix=True
        if values["K_matrix"] == False:
            matrix_global= np.zeros((2,2))
        if values["Local_K_matrix"] == False:
            local_K_matrix=False
        if values["Deformations"] == False:
            deformations= np.zeros((2,2))
        if values["reactions"] == False:
            nodal_reactions= np.zeros((2,2))
        if values["nodal_stresses"] == False:
            nodal_stresses= np.zeros((2,2))
        Printing_results (number_elements,number_nodes,values["-OUT-"],matrix_global,local_K_matrix,deformations,nodal_reactions,von_mises_stress,nodal_stresses)
_VARS['window'].close()
