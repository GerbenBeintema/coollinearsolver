from scipy.io import loadmat
from cool_linear_solver import Variable, System_of_linear_eqs
import numpy as np


test = False
if test:
    A = np.array([[-3, 0, 0],[0, -1, 0], [1/4, -1/3, -1/2]])
    B = np.array([[2, 2, 1]]).T
    C = np.array([[0, 1, 2]])
else:
    system = loadmat("./largematrixes.mat") #48 by 48
    A, B, C, D = system["A"], system["B"], system["C"], system["D"]

N = A.shape[0]
p = Variable('p')
P = [[p[i,j] if j<=i else p[j,i] for j in range(N)] for i in range(N)] #create symetric matrix
P = np.array(P, dtype=object) #create numpy array such to use matrix multiply

controlability = True
if controlability: 
    G = A@P + P@A.T + B@B.T
else:
    G = A.T@P + P@A + C.T@C

print('G:',G.shape)

#add equation so a system
sys = System_of_linear_eqs()
for i in range(N):
    for j in range(N):
        if i<=j: #add only lower triangular terms
            sys.add_equation(G[i,j]==0)

sys.solve() #solve

M = sys.get_sparse_matrix() #look at sparse matrix
print('M:',M.__repr__(),'sparse matrix has been created')
print(f'Fullness: {M.nnz/(M.shape[0]*M.shape[1]):.5%}') # a sparse matrix is automaticly created

P = np.array([[sys[p[i,j] if j<=i else p[j,i]] for j in range(N)] for i in range(N)]) #extract solution from solution

print('P',P.shape,'\n',P)
print('P eigenvalues:\n',np.linalg.eigvals(P))

R = np.linalg.cholesky(P)
print('R',R.shape,'\n',R)
from matplotlib import pyplot as plt
print('R eigenvalues:\n',np.linalg.eigvals(R))

######### visualize #############
plt.figure(figsize=(15,8))
plt.subplot(2,4,1)
plt.imshow(A)
plt.title('A')
plt.subplot(2,4,2)
if controlability:
    plt.imshow(B@B.T)
    plt.title('B@B.T')
else:
    plt.imshow(C.T@C)
    plt.title('C.T@C')

plt.subplot(2,4,3)
plt.imshow(P)
if controlability:
    plt.title('P of A@P + P@A.T + B@B.T=0')
else:
    plt.title('P of A.T@P + P@A + C.T@C = 0')
plt.subplot(2,4,4)
plt.imshow(R)
plt.title('R = cholesky of P')

plt.subplot(2,4,5)
plt.semilogy(abs(np.linalg.eigvals(A)))
plt.title('eigen values |A|')
plt.subplot(2,4,6)
if controlability:
    plt.semilogy(np.linalg.eigvals(B@B.T))
    plt.title('eigen values B@B.T')
else:
    plt.semilogy(np.linalg.eigvals(C.T@C))
    plt.title('eigen values C.T@C')

plt.subplot(2,4,7)
plt.semilogy(np.linalg.eigvals(P))
plt.title('eigen values P')
plt.subplot(2,4,8)
plt.semilogy(sorted(np.linalg.eigvals(R),reverse=True))
plt.title('eigen values R')
plt.show()
