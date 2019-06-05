import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt
from scipy import linalg as la
import time


m = 100
r = m
n = m

m1 = 20
m2 = m1
b =  m1*m2
d = 50


np.random.seed(4)
A = np.random.uniform(0,10,(m,r))
#print(A)
#B = np.random.randint(0,10,(r,n))
id1 = 1
id2 = 8
temp = A[id1,:]
A[id2,:] = temp
B = np.copy(np.transpose(A))



m,r = A.shape
n = B.shape[1]
print('b', b)
print('ratio:',m*n/float(b))
M = max(m,n)
p = np.zeros((d,b))
p_real = np.zeros((d,b))
p_imag = np.zeros((d,b))
PA_cs = np.zeros((r,d,b))
s1 = np.zeros((d,M))
h1 = np.zeros((d,M))
s2 = np.zeros((d,M))
h2 = np.zeros((d,M))



t1 = time.time()
for num in range(d):
    #s1[num] = 2*np.random.randint(0,2,(M,))-1
    #h1[num] = np.random.randint(0,b,(M,))
    np.random.seed(num)
    h1[num] = np.random.choice(b, M, replace=True)
    np.random.seed(num)
    s1[num] = np.random.choice(2, M, replace=True) * 2 - 1
    np.random.seed(num+1)
    h2[num] = np.random.choice(b, M, replace=True)
    np.random.seed(num+1)
    s2[num] = np.random.choice(2, M, replace=True) * 2 - 1
    #print(s1[num])
    #print(h1[num])
    #print(s2[num])
    #print(h2[num])
    for k in range(r):
        pa = np.zeros((b,1))
        pb = np.zeros((b,1))
        for i in range(m):
            pa[int(h1[num][i])] += s1[num][i]*A[i][k]
        for j in range(n):
            pb[int(h2[num][j])] += s2[num][j]*B[k][j]
        PA_cs[k,num,:] = np.reshape(pa,(b,))
        #print('ha:',h1[num])
        #print('before:',pa)
        pa = np.fft.fft(pa,axis = 0)
        pb = np.fft.fft(pb,axis = 0)
        p_real[num]+= np.reshape(np.real(pa)*np.real(pb),(b,))
        p_imag[num]+= np.reshape(np.imag(pa)*np.imag(pb),(b,))
        p_temp = p_real[num,:]+1j*p_imag[num,:]  
    p[num] = np.fft.ifft(p_temp)
    #print('pnum:',p[num])
t2 = time.time()

def decompress(p,s1,h1,s2,h2,d,b,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*s2[num][j]*p[num][int(h1[num][i]+h2[num][j])% b]
    #print(x)
    return np.median(x)

def decompress_single(p,s1,h1,d,b,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*p[num][(h1[num][i])% b]
    #print(x)
    return np.median(x)


c = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        c[i][j] = decompress(p,s1,h1,s2,h2,d,b,i,j)
'''
A_cs = np.zeros((m,r))
for i in range(m):
    for j in range(r):
        paa = PA_cs[j,:,:]
        A_cs[i][j] = decompress_single(paa,s1,h1,d,b,i,j)
'''

#print('AA:',AA)
#print('A:',A)
#print('diff:',LA.norm(A-AA, 'fro')/LA.norm(A, 'fro'))
#print("estimate")
#print(c)
#print("original")
C = np.dot(A,B)
#print(C)
#plt.imshow(C);
#plt.show()
#f = plt.figure(1)
#plt.imshow(c);
#plt.colorbar()





#g = plt.figure(2)




#axarr[1, 1].imshow(np.dot(AA,np.transpose(AA)), cmap="binary")
#axarr[1, 1].set_title('Axis [1,1]')


#plt.imshow(c_copy);
#plt.colorbar()
           
diff = C-c
n_dif = LA.norm(diff, 'fro')
print('cs:')
print(n_dif/LA.norm(C,'fro'))

#print(c[id1][id2]-C[id1][id2])
print('element:')
print(abs(c[id1][id2]-C[id1][id2])/C[id1][id2])
print('time:')
print((t2-t1)/d)


##############MTS###################

def decompress_mts(p,s1,h1,s2,h2,d,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*s2[num][j]*p[num,int(h1[num][i]), int(h2[num][j])]
    #print(x)
    return np.median(x)



m,r = A.shape
n = B.shape[1]
print('m1', m1)
print('ratio:',m*n/float(m1*m2))
p = np.zeros((d,m1,m2))
s1 = np.zeros((d,m))
h1 = np.zeros((d,m))
s2 = np.zeros((d,n))
h2 = np.zeros((d,n))

t1 = time.time()
for num in range(d):
    #s1[num] = 2*np.random.randint(0,2,(M,))-1
    #h1[num] = np.random.randint(0,b,(M,))
    np.random.seed(num+3)
    h1[num] = np.random.choice(m1, m, replace=True)
    np.random.seed(num+3)
    s1[num] = np.random.choice(2, m, replace=True) * 2 - 1
    np.random.seed(num*2+1)
    h2[num] = np.random.choice(m2, n, replace=True)
    np.random.seed(num*2+1)
    s2[num] = np.random.choice(2, n, replace=True) * 2 - 1
    H1 = np.zeros((m1,m))
    H2 = np.zeros((m2,n))
    for i in range(m):
        H1[h1[num][i]][i] = 1
    for i in range(n):
        H2[h2[num][i]][i] = 1
    S = np.outer(s1[num],s2[num])
    p[num,:,:] =  np.dot(np.dot(H1,S*np.dot(A,B)),np.transpose(H2))
t2 = time.time()

c_mts = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        c_mts[i][j] = decompress_mts(p,s1,h1,s2,h2,d,i,j)

diff = C-c_mts
n_dif = LA.norm(diff, 'fro')
print('mts:')
print(n_dif/LA.norm(C,'fro'))
print('element:')
print(abs(c_mts[id1][id2]-C[id1][id2])/C[id1][id2])
print('time:')
print((t2-t1)/d)


'''
f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(A)
axarr[0, 0].set_title(r'$A$',fontsize =25)
axarr[0, 1].imshow(C)
axarr[0, 1].set_title(r'$A A^T$',fontsize=25)
axarr[1, 0].imshow(c)
axarr[1, 0].set_title(r'$A A^T_{ts}$',fontsize=25)
axarr[1, 1].imshow(c_mts)
axarr[1, 1].set_title(r'$AA^T_{mts}$',fontsize=25)
plt.show() 

'''




'''

############ MS #############
n1,n2 = A.shape
n3,n4 = B.shape
if n2 == n3:
    r = n2

print('m1:',m1)
print('m2:',m2)
print('ratio:',m*n*r*r/float(m1*m2))
M1 = max(n1,n3)
M2 = max(n2,n4)
P = np.zeros((d,m1,m2))
PA_ms = np.zeros((d,m1,m2))
s1 = np.zeros((d,M1))
h1 = np.zeros((d,M1))
s2 = np.zeros((d,M2))
h2 = np.zeros((d,M2))
for num in range(d):
    #s1[num] = 2*np.random.randint(0,2,(M,))-1
    #h1[num] = np.random.randint(0,b,(M,))
    np.random.seed(d)
    h1[num] = np.random.choice(m1, M1, replace=True)
    np.random.seed(d)
    s1[num] = np.random.choice(2, M1, replace=True) * 2 - 1
    np.random.seed(d*2+1)
    s2[num] = np.random.choice(2, M2, replace=True) * 2 - 1
    np.random.seed(d*2+1)
    h2[num] = np.random.choice(m2, M2, replace=True)
    #print(s1[num])
    #print(h1[num])
    #print(s2[num])
    #print(h2[num])
   
    pa = np.zeros((m1,m2))
    pb = np.zeros((m1,m2))
    for i in range(n1):
        for j in range(n2):
            pa[int(h1[num][i]),int(h2[num][j])] += s1[num][i]*s2[num][j]*A[i][j]
    for i in range(n3):
        for j in range(n4):
            pb[int(h1[num][i]),int(h2[num][j])] += s1[num][i]*s2[num][j]*B[i][j]
        
    PA_ms[num] = pa
    pa = np.fft.fft2(pa)
    pb = np.fft.fft2(pb)
    p_real = np.real(pa)*np.real(pb)
    p_imag = np.imag(pa)*np.imag(pb)
    p = p_real+1j*p_imag
    #P[num,:,:]= np.real(np.fft.ifft2(np.reshape(pa*pb,(m1,m2,))))
    P[num,:,:]= np.real(np.fft.ifft2(p))


def decompress_ms(P,s1,h1,s2,h2,d,n1,n2,n3,n4):
    Q = np.zeros((n1*n3,n2*n4))
    temp = np.zeros((d,1))
    
    for p in range(n1):
        for q in range(n2):
            for h in range(n3):
                for g in range(n4):
                    for num in range(d):
                        temp[num] = s1[num][p]*s2[num][q]*s1[num][h]*s2[num][g]*P[num][(h1[num][p]+h1[num][h])% m1][(h2[num][q]+h2[num][g])% m2]
                    i = n3*(p+1-1)+h+1
                    j = n4*(q+1-1)+g+1
                    Q[i-1,j-1] = np.median(temp)
    return Q


def decompress_single_ms(p,s1,h1,s2,h2,d,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*s2[num][j]*p[num][h1[num][i],h2[num][j]]
    #print(x)
    return np.median(x)

'''


'''
A_ms = np.zeros((n1,n2))
for i in range(n1):
    for j in range(n2):
        A_ms[i][j] = decompress_single_ms(PA_ms,s1,h1,s2,h2,d,i,j)
'''


'''
Q_ms = decompress_ms(P,s1,h1,s2,h2,d,n1,n2,n3,n4)
print(Q_ms.shape)

QQ = np.kron(A,B)
diff = QQ-Q_ms
n_dif_ms = LA.norm(diff, 'fro')
print('ms:')
print(n_dif_ms/LA.norm(QQ,'fro'))



c_ms = np.zeros((n1,n4))
for i in range(n1):
    for j in range(n4):
        for k in range(r):
            c_ms[i][j] += Q_ms[n3*(i+1-1)+k+1-1][n4*(k+1-1)+j+1-1]



diff = C-c_ms
n_dif_ms = LA.norm(diff, 'fro')
print('ms:')
print(n_dif_ms/LA.norm(C,'fro'))
#print(c_ms[id1][id2]-C[id1][id2])
#print((c_ms[id1][id2]-C[id1][id2])/C[id1][id2])

'''

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