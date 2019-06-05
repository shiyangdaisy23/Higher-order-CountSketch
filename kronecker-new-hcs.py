import numpy as np
from numpy import linalg as LA
#import matplotlib.pylab as plt
from scipy import linalg as la
import time
from memory_profiler import profile

m = 30
#m = 10
r = m
n = m

d = 20
#sketch_list = [5,10,15,40,100]
#sketch_list = [50,100,500,1000]
#sketch_list = [80,200,800,2000,4000]
#sketch_list = [100,1000,5000,8000]

m1 = 70
m2 = m1
#sketch_list = [10,14,17,22, 31, 70]

#sketch_list = [31]
#res_ms = np.zeros((len(sketch_list),1))
#time_ms = np.zeros((len(sketch_list),1))
############ input A ##############
np.random.seed(4)
A = np.random.uniform(-5,5,(m,r))
np.random.seed(4+1)
B = np.random.uniform(-5,5,(r,n))
'''
id1 = 1
id2 = 3
temp = A[id1,:]
A[id2,:] = temp
B = np.copy(np.transpose(A))
'''
QQ = np.kron(A,B)
###############################
def decompress_ms(P,s1,h1,s2,h2,d,n1,n2,n3,n4):
    Q = np.zeros((n1*n3,n2*n4))
    temp = np.zeros((d,1))
    
    for p in range(n1):
        #print(p,'/',n1)
        for q in range(n2):
            for h in range(n3):
                for g in range(n4):
                    for num in range(d):
                        temp[num] = s1[num][p]*s2[num][q]*s1[num][h]*s2[num][g]*P[num][(int(h1[num][p])+int(h1[num][h]))% m1][(int(h2[num][q])+int(h2[num][g]))% m2]
                    i = n3*(p)+h
                    j = n4*(q)+g
                    Q[i,j] = np.median(temp)
    return Q

####################################
@profile
def main_compress(m1,m2,M1,M2,A,B,n1,n2,n3,n4):

    P = np.zeros((d,m1,m2))
    s1 = np.zeros((d,M1))
    h1 = np.zeros((d,M1))
    s2 = np.zeros((d,M2))
    h2 = np.zeros((d,M2))
    s3 = np.zeros((d,M1))
    h3 = np.zeros((d,M1))
    s4 = np.zeros((d,M2))
    h4 = np.zeros((d,M2))
    
    for num in range(d):
        np.random.seed(num)
        h1[num] = np.random.choice(m1, M1, replace=True)
        np.random.seed(num)
        s1[num] = np.random.choice(2, M1, replace=True) * 2 - 1
        np.random.seed(num+1)
        s2[num] = np.random.choice(2, M2, replace=True) * 2 - 1
        np.random.seed(num+1)
        h2[num] = np.random.choice(m2, M2, replace=True)
        np.random.seed(num+3)
        h3[num] = np.random.choice(m1, M1, replace=True)
        np.random.seed(num+3)
        s3[num] = np.random.choice(2, M1, replace=True) * 2 - 1
        np.random.seed(num+10)
        s4[num] = np.random.choice(2, M2, replace=True) * 2 - 1
        np.random.seed(num+10)
        h4[num] = np.random.choice(m2, M2, replace=True)

        H1 = np.zeros((m1,n1))
        H2 = np.zeros((m2,n2))
        H3 = np.zeros((m1,n3))
        H4 = np.zeros((m2,n4))
        for i in range(n1):
            H1[h1[num][i]][i] = 1
        for i in range(n2):
            H2[h2[num][i]][i] = 1
        for i in range(n3):
            H3[h3[num][i]][i] = 1
        for i in range(n4):
            H4[h4[num][i]][i] = 1
        


        S1 = np.outer(s1[num],s2[num])
        S2 = np.outer(s3[num],s4[num])
        
        tempA = np.tensordot(S1*A,H1, axes=([0],[1]))
        pa = np.tensordot(tempA,H2, axes=([0],[1]))
        tempB = np.tensordot(S2*B,H3, axes=([0],[1]))
        pb = np.tensordot(tempB,H4, axes=([0],[1]))

        pa = np.fft.fft2(pa)
        pb = np.fft.fft2(pb)
        p_real = np.real(pa)*np.real(pb)
        p_imag = np.imag(pa)*np.imag(pb)
        p = p_real+1j*p_imag
        #P[num,:,:]= np.real(np.fft.ifft2(np.reshape(pa*pb,(m1,m2,))))
        P[num,:,:]= np.real(np.fft.ifft2(p))
    return P,h1,s1,h2,s2
############ MS #############
n1,n2 = A.shape
n3,n4 = B.shape

print('m1:',m1)
print('m2:',m2)
print('ratio:',n1*n2*n3*n4/float(m1*m2))
M1 = max(n1,n3)
M2 = max(n2,n4)

t1 = time.time()
P,h1,s1,h2,s2 = main_compress(m1,m2,M1,M2,A,B,n1,n2,n3,n4)
t2 = time.time()

print('Decompress now')


Q_ms = np.zeros((n1*n3,n2*n4))
temp = np.zeros((d,1))
t3 = time.time()    
for p in range(n1):
    #print(p,'/',n1)
    for q in range(n2):
        for h in range(n3):
            for g in range(n4):
                for num in range(d):
                    temp[num] = s1[num][p]*s2[num][q]*s1[num][h]*s2[num][g]*P[num][int(h1[num][p]+h1[num][h])% m1][int(h2[num][q]+h2[num][g])% m2]
                i = n3*(p)+h
                j = n4*(q)+g
                Q_ms[int(i),int(j)] = np.median(temp)
#Q_ms = decompress_ms(P,s1,h1,s2,h2,d,n1,n2,n3,n4)
t4 = time.time()

diff = QQ-Q_ms
n_dif_ms = LA.norm(diff, 'fro')
print('ms:')
print(n_dif_ms/LA.norm(QQ,'fro'))
print('compress time:')
print((t2-t1))
print('decompress time:')
print((t4-t3))

'''
f, axarr = plt.subplots(2, 3)
f.patch.set_facecolor('white')
axarr[0, 0].imshow(A)
axarr[0, 0].set_title(r'$A$',fontsize=25)
axarr[0, 1].imshow(C)
axarr[0, 1].set_title(r'$AA^T$',fontsize=25)
axarr[0, 2].imshow(c)
axarr[0, 2].set_title(r'$AA^T_{cs}$',fontsize=25)
axarr[1, 0].imshow(np.kron(A,B))
axarr[1, 0].set_title(r'$A \otimes A^T$',fontsize=25)
axarr[1, 1].imshow(Q_ms)
axarr[1, 1].set_title(r'$A \otimes A^T_{ms}$',fontsize=25)
axarr[1, 2].imshow(c_ms)
axarr[1, 2].set_title(r'$AA^T_{ms}$',fontsize=25)
plt.show()
'''
