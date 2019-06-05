import numpy as np
from numpy import linalg as LA
#import matplotlib.pylab as plt
from scipy import linalg as la
import time
from memory_profiler import profile

m = 30
r = m
n = m

d = 20
#sketch_list = [1,4,9,64,500]
#sketch_list = [4,9,25,81,144]
#sketch_list = [50,100,500,1000]
#sketch_list = [80,200,800,2000,4000]
b = 5000
#100,200,300,500, 1000, 5000]
#sketch_list = [1000]

#res_cs = np.zeros((len(sketch_list),1))
#time_cs = np.zeros((len(sketch_list),1))
############ input #########
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
##################################
def decompress_new(p,s1,h1,s2,h2,d,b,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*s2[num][j]*p[num][int(h1[num][i]+h2[num][j])% b]
    return np.median(x)

##################################
def decompress(p,s1,h1,d,b,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*s1[num][j]*p[num][int(h1[num][i]+h1[num][j])% b]
    return np.median(x)



##################################
@profile
def main_compress(b,d,M,A,B,n1,n2,n3,n4):
    p = np.zeros((d,b))
    p_real = np.zeros((b,1))
    p_imag = np.zeros((b,1))
    s1 = np.zeros((d,M))
    h1 = np.zeros((d,M))
    s2 = np.zeros((d,M))
    h2 = np.zeros((d,M))
    for num in range(d):
        np.random.seed(num)
        h1[num] = np.random.choice(b, M, replace=True)
        np.random.seed(num)
        s1[num] = np.random.choice(2, M, replace=True) * 2 - 1
        np.random.seed(num+1)
        h2[num] = np.random.choice(b, M, replace=True)
        np.random.seed(num+1)
        s2[num] = np.random.choice(2, M, replace=True) * 2 - 1
        
        '''
        pa = np.zeros((b,1))
        pb = np.zeros((b,1))
        for i1 in range(n1):
            for i2 in range(n2):
                l = n2*i1+i2
                pa[int(h1[num][l])] += s1[num][l]*A[i1][i2]
        for i1 in range(n3):
            for i2 in range(n4):
                l = n4*i1+i2
                pb[int(h2[num][l])] += s2[num][l]*B[i1][i2]
        '''

        H1 = np.zeros((b,M))
        H2 = np.zeros((b,M))
        for i in range(n1):
            for j in range(n2):
                H1[h1[num][i*n2+j]][i*n2+j] = 1
        for i in range(n3):
            for j in range(n4):
                H2[h2[num][i*n4+j]][i*n4+j] = 1

        pa = np.dot(H1,s1[num]*np.reshape(A,(n1*n2,)))
        pb = np.dot(H2,s2[num]*np.reshape(B,(n3*n4,)))
        pa = np.fft.fft(pa,axis = 0)
        pb = np.fft.fft(pb,axis = 0)
        p_real = np.reshape(np.real(pa)*np.real(pb),(b,))
        p_imag = np.reshape(np.imag(pa)*np.imag(pb),(b,))
        p_temp = p_real+1j*p_imag
        p[num,:] = np.fft.ifft(p_temp)
    return p,h1,s1,h2,s2


############ cs #############


n1,n2 = A.shape
n3,n4 = B.shape
    
M = max(n1*n2,n3*n4)
print('b', b)
print('ratio:',m*n*r**2/float(b))

t1 = time.time()
p,h1,s1,h2,s2 = main_compress(b,d,M,A,B,n1,n2,n3,n4)

t2 = time.time()

Q_cs = np.zeros((n1*n3,n2*n4))
t3 = time.time()
for i1 in range(n1):
    for i2 in range(n2):
        for i3 in range(n3):
            for i4 in range(n4):
                i = n2*i1+i2
                j = n4*i3+i4
                c1 = n3*i1+i3
                c2 = n4*i2+i4
                Q_cs[c1][c2] = decompress_new(p,s1,h1,s2,h2,d,b,i,j)  
t4 = time.time()            
    
diff = QQ-Q_cs
n_dif_cs = LA.norm(diff, 'fro')
print('ts:')
print(n_dif_cs/LA.norm(QQ,'fro'))
#res_cs[idx] = n_dif_cs/LA.norm(QQ,'fro')
#time_cs[idx] = (t2-t1)
print('compress time:')
print((t2-t1))
print('decompress time:')
print((t4-t3))
'''
np.save('error-ts10.npy',res_cs)
np.save('time-ts10.npy',time_cs)
'''
