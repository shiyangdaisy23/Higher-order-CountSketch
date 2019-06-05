import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt
from scipy import linalg as la
import time
from memory_profiler import profile

#AB, A \in m*m*r, B \in r*m*m
m = 30
r = 40
n = m

m1 = 25
m2 = m1
b =  m1*m2*m1*m2
d = 20


np.random.seed(4)
A = np.random.uniform(0,10,(m,m,r))
#print(A)
#B = np.random.randint(0,10,(r,n))

B = np.random.uniform(0,10,(r,m,m))

C = np.tensordot(A,B, axes=([2],[0]))
print(C.shape)




@profile
def main_cs(d,b,m,A,B,p):
    M = m**2
    
    p_real = np.zeros((b,1))
    p_imag = np.zeros((b,1))

    s1 = np.zeros((d,M))
    h1 = np.zeros((d,M))
    s2 = np.zeros((d,M))
    h2 = np.zeros((d,M))

    H1 = np.zeros((b,m**2))
    H2 = np.zeros((b,m**2))
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
        

        H1 = np.zeros((b,m**2))
        H2 = np.zeros((b,m**2))
        for i in range(m):
            for j in range(m):
                H1[h1[num][i*m+j]][i*m+j] = 1
        for i in range(m):
            for j in range(m):
                H2[h2[num][i*m+j]][i*m+j] = 1

        for k in range(r):
            #PA_cs[k,num,:] = np.reshape(pa,(b,))
            #print('ha:',h1[num])
            #print('before:',pa)
            pa = np.dot(H1,s1[num]*np.reshape(A[:,:,k],(m*m,)))
            pb = np.dot(H2,s2[num]*np.reshape(B[k,:,:],(m*m,)))
            '''
            patest = np.zeros((b,1))
            for test in range(m):
                for test2 in range(m):
                    patest[h1[num][test*m+test2]] += s1[num][test*m+test2]*A[test,test2,k]
            pbtest = np.zeros((b,1))
            for test in range(m):
                for test2 in range(m):
                    pbtest[h2[num][test*m+test2]] += s2[num][test*m+test2]*B[k,test,test2]
            #print(patest)
            print(k)
            print(np.nonzero(np.reshape(pa,(b,1))-patest))
            print(np.nonzero(np.reshape(pb,(b,1))-pbtest))
            '''
            pa1 = np.fft.fft(pa,axis = 0)
            pb1 = np.fft.fft(pb,axis = 0)
            p_real= np.reshape(np.real(pa1)*np.real(pb1),(b,))
            p_imag= np.reshape(np.imag(pa1)*np.imag(pb1),(b,))
            p_temp = p_real+1j*p_imag  
            p[num] += np.fft.ifft(p_temp)
    return p,h1,h2,s1,s2

print('b', b)
print('ratio:',m**4/float(b))
p = np.zeros((d,b))

t1 = time.time()
p,h1,h2,s1,s2 = main_cs(d,b,m,A,B,p)
t2 = time.time()


def decompress(p,s1,h1,s2,h2,d,b,i,j,k,l):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i*m+j]*s2[num][k*m+l]*p[num][int(h1[num][i*m+j]+h2[num][k*m+l])% b]
    #print(x)
    return np.median(x)

def decompress_single(p,s1,h1,d,b,i,j):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*p[num][(h1[num][i])% b]
    #print(x)
    return np.median(x)

t3 = time.time()
c = np.zeros((m,m,m,m))
for i in range(m):
    for j in range(m):
        for k in range(m):
            for l in range(m):
                c[i][j][k][l] = decompress(p,s1,h1,s2,h2,d,b,i,j,k,l)
t4 = time.time()

           
diff = C-c
diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('cs:')
print(n_dif/LA.norm(np.reshape(C,(m*m,m*m)),'fro'))

print('compress time:')
print(t2-t1)
print('decompress time:')
print(t4-t3)

##############MTS###################
'''
def decompress_mts(p,s1,h1,s2,h2,s3,h3,s4,h4,d,i,j,k,l):
    x = np.zeros((d,1))
    for num in range(d):
        x[num] = s1[num][i]*s2[num][j]*s3[num][k]*s4[num][l]*p[num,h1[num][i], h2[num][j],h3[num][k], h4[num][l]]
    #print(x)
    return np.median(x)

@profile
def main_hs(m,m1,m2,d,A,B,p):

    s1 = np.zeros((d,m))
    h1 = np.zeros((d,m))
    s2 = np.zeros((d,m))
    h2 = np.zeros((d,m))
    s3 = np.zeros((d,m))
    h3 = np.zeros((d,m))
    s4 = np.zeros((d,m))
    h4 = np.zeros((d,m))


    for num in range(d):
        #s1[num] = 2*np.random.randint(0,2,(M,))-1
        #h1[num] = np.random.randint(0,b,(M,))
        np.random.seed(num+3)
        h1[num] = np.random.choice(m1, m, replace=True)
        np.random.seed(num+3)
        s1[num] = np.random.choice(2, m, replace=True) * 2 - 1
        np.random.seed(num*2+1)
        h2[num] = np.random.choice(m2, m, replace=True)
        np.random.seed(num*2+1)
        s2[num] = np.random.choice(2, m, replace=True) * 2 - 1

        h3[num] = np.random.choice(m1, m, replace=True)
        np.random.seed(num+8)
        s3[num] = np.random.choice(2, m, replace=True) * 2 - 1
        np.random.seed(num*2+8)
        h4[num] = np.random.choice(m2, m, replace=True)
        np.random.seed(num*2+8)
        s4[num] = np.random.choice(2, m, replace=True) * 2 - 1

        H1 = np.zeros((m1,m))
        H2 = np.zeros((m2,m))
        H3 = np.zeros((m1,m))
        H4 = np.zeros((m2,m))

        for i in range(m):
            H1[h1[num][i]][i] = 1
        for i in range(m):
            H2[h2[num][i]][i] = 1
        for i in range(m):
            H3[h3[num][i]][i] = 1
        for i in range(m):
            H4[h4[num][i]][i] = 1
        S1 = np.outer(s1[num],s2[num])
        S2 = np.outer(s3[num],s4[num])
        S = np.tensordot(S1,S2,axes = 0)
        #print(S.shape)
        temppp = np.tensordot(S*C,H1, axes=([0],[1]))
        temppp = np.tensordot(temppp,H2, axes=([0],[1]))
        temppp = np.tensordot(temppp,H3, axes=([0],[1]))
        p[num,:,:,:,:] = np.tensordot(temppp,H4, axes=([0],[1])) 
    return p,H1,H2,H3,H4,S

print('ratio:',m**4/float(m1*m2*m1*m2))
p = np.zeros((d,m1,m2,m1,m2))
t5 = time.time()
p,H1,H2,H3,H4,S = main_hs(m,m1,m2,d,A,B,p)
t6 = time.time()




t71 = time.time()
c_mts_num = np.zeros((d,m,m,m,m))
c_mts = np.zeros((m,m,m,m))
for num in range(d):
    pnum = p[num,:,:,:,:]
    tempp = np.tensordot(pnum,np.transpose(H1), axes=([0],[1]))
    tempp = np.tensordot(tempp,np.transpose(H2), axes=([0],[1]))
    tempp = np.tensordot(tempp,np.transpose(H3), axes=([0],[1]))
    c_mts_num[num,:,:,:,:] = S * np.tensordot(tempp,np.transpose(H4), axes=([0],[1]))

for i in range(m):
    for j in range(m):
        for k in range(m):
            for l in range(m):
                c_mts[i][j][k][l] = np.median(c_mts_num[:,i,j,k,l])

t81 = time.time()

'''
'''
t7 = time.time()
c_mts_naive = np.zeros((m,m,m,m))
for i in range(m):
    print(i)
    for j in range(m):
        for k in range(m):
            for l in range(m):
                c_mts_naive[i][j][k][l] = decompress_mts(p,s1,h1,s2,h2,s3,h3,s4,h4,d,i,j,k,l)
t8 = time.time()
'''
'''

diff = C-c_mts
diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('mts:')
print(n_dif/LA.norm(np.reshape(C,(m*m,m*m)),'fro'))



diff = C-c_mts_naive
diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('mts_naive:')
print(n_dif/LA.norm(np.reshape(C,(m*m,m*m)),'fro'))


print('compress time:')
print((t6-t5)/d)
print('decompress time:')
print((t81-t71)/d)
#print('naive decompress time:')
#print((t8-t7)/d)
'''