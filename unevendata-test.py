import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt
from scipy import linalg as la
import time
from memory_profiler import profile

#sketch A \in \R^{n*n}

n = 50
d = 2
m1 = 2
m2 = m1
b = m1*m2
print('compression ratio',n*n/(b))




np.random.seed(4)
A = np.random.uniform(-1,1,(n,n))


for i in range(n):
	A[i,2] = 100
#print(A)



print('not even')
#print('CS compress now')

@profile
def CS_main(d,n,b,A):
	s1 = np.zeros((d,n*n))
	h1 = np.zeros((d,n*n))
	p = np.zeros((d,b))

	for num in range(d):
		np.random.seed(num)
		h1[num] = np.random.choice(b, n*n, replace=True)
		np.random.seed(num)
		s1[num] = np.random.choice(2,n*n, replace=True) * 2 - 1
	
		H1 = np.zeros((b,n*n))

		for i in range(n):
			for j in range(n):
				H1[h1[num][i*n+j]][i*n+j] = 1
		#print((s1[num]*np.reshape(A,(n*n,))).shape)
		p[num,:] = np.tensordot(s1[num]*np.reshape(A,(n*n,)),H1, axes=([0],[1]))
	return p,s1,h1
'''
t1= time.time()
p,s1,h1 = CS_main(d,n,b,A)
t2= time.time()

#recovery
#print('CS recovery now')
A_cs_all = np.zeros((d,n*n))
t3 = time.time()
for num in range(d):
	H1 = np.zeros((b,n*n))

	for i in range(n):
		for j in range(n):
			H1[h1[num][i*n+j]][i*n+j] = 1


	A_cs_all[num,:] = s1[num]*np.tensordot(p[num,:],np.transpose(H1), axes=([0],[1]))
	
A_cs = np.zeros((n,n))   
for i in range(n):
	for j in range(n):
		A_cs[i,j] = np.median(A_cs_all[:,i*n+j])
t4 = time.time()

diff = A-A_cs
#diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('cs:')
print(n_dif/LA.norm(A,'fro'))
print('compress time:', t2-t1)
print('decompress time:', t4-t3)

exit()
'''
########### HCS ##################
@profile
def HCS_main(d,n,m1,m2,A):

	s1 = np.zeros((d,n))
	h1 = np.zeros((d,n))
	s2 = np.zeros((d,n))
	h2 = np.zeros((d,n))
	P = np.zeros((d,m1,m2))

	#print('HCS compress now')
	for num in range(d):
		np.random.seed(num)
		h1[num] = np.random.choice(m1, n, replace=True)
		np.random.seed(num)
		s1[num] = np.random.choice(2,n, replace=True) * 2 - 1
		np.random.seed(num+1)
		s2[num] = np.random.choice(2, n, replace=True) * 2 - 1
		np.random.seed(num+1)
		h2[num] = np.random.choice(m2, n, replace=True)
		#print(h1[num])
		#print(h2[num])
		H1 = np.zeros((m1,n))
		H2 = np.zeros((m2,n))

		for i in range(n):
			H1[h1[num][i]][i] = 1
		for i in range(n):
			H2[h2[num][i]][i] = 1
		S = np.outer(s1[num],s2[num])
		#print(S)
		tempA = np.tensordot(S*A,H1, axes=([0],[1]))
		P[num,:,:] = np.tensordot(tempA,H2, axes=([0],[1]))
		#print(P[num,:,:])
	return P,s1,h1,s2,h2

'''
t1 = time.time()
P,s1,h1,s2,h2 = HCS_main(d,n,m1,m2,A)
t2= time.time()
#recovery
#print('HCS recovery now')
A_mts_all = np.zeros((d,n,n))
t3 = time.time()
for num in range(d):
	H1 = np.zeros((m1,n))
	H2 = np.zeros((m2,n))

	for i in range(n):
	    H1[h1[num][i]][i] = 1
	for i in range(n):
	    H2[h2[num][i]][i] = 1
	S = np.outer(s1[num],s2[num])

	tempp = np.tensordot(P[num,:,:],np.transpose(H1), axes=([0],[1]))
	A_mts_all[num,:,:] = S*np.tensordot(tempp,np.transpose(H2), axes=([0],[1]))
#print(A_mts_all)
A_mts = np.zeros((n,n))   
for i in range(n):
	for j in range(n):
		A_mts[i,j] = np.median(A_mts_all[:,i,j])
t4= time.time()

diff = A-A_mts
#diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('hcs:')
print(n_dif/LA.norm(A,'fro'))
print('compress time:', t2-t1)
print('decompress time:', t4-t3)

exit()

'''

print('even!')
for i in range(n):
	A[i,i],A[i,2] = A[i,2],A[i,i]
#print(A)


'''

t1= time.time()
p,s1,h1 = CS_main(d,n,b,A)
t2= time.time()


#recovery
#print('CS recovery now')
A_cs_all = np.zeros((d,n*n))
t3 = time.time()
for num in range(d):
	H1 = np.zeros((b,n*n))

	for i in range(n):
		for j in range(n):
			H1[h1[num][i*n+j]][i*n+j] = 1


	A_cs_all[num,:] = s1[num]*np.tensordot(p[num,:],np.transpose(H1), axes=([0],[1]))
	
A_cs = np.zeros((n,n))   
for i in range(n):
	for j in range(n):
		A_cs[i,j] = np.median(A_cs_all[:,i*n+j])
t4= time.time()

diff = A-A_cs
#diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('cs:')
print(n_dif/LA.norm(A,'fro'))
print('compress time:', t2-t1)
print('decompress time:', t4-t3)

exit()
'''
########### HCS ##################
t1 = time.time()
P,s1,h1,s2,h2 = HCS_main(d,n,m1,m2,A)
t2= time.time()



#recovery
#print('HCS recovery now')
A_mts_all = np.zeros((d,n,n))
t3 = time.time()
for num in range(d):
	H1 = np.zeros((m1,n))
	H2 = np.zeros((m2,n))

	for i in range(n):
	    H1[h1[num][i]][i] = 1
	for i in range(n):
	    H2[h2[num][i]][i] = 1
	S = np.outer(s1[num],s2[num])

	tempp = np.tensordot(P[num,:,:],np.transpose(H1), axes=([0],[1]))
	A_mts_all[num,:,:] = S*np.tensordot(tempp,np.transpose(H2), axes=([0],[1]))
#print(A_mts_all)
A_mts = np.zeros((n,n))   
for i in range(n):
	for j in range(n):
		A_mts[i,j] = np.median(A_mts_all[:,i,j])
t4 = time.time()

diff = A-A_mts
#diff = np.reshape(diff,(m*m,m*m))
n_dif = LA.norm(diff, 'fro')
print('hcs:')
print(n_dif/LA.norm(A,'fro'))
print('compress time:', t2-t1)
print('decompress time:', t4-t3)






