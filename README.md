# STA141C_Final_Project
%%time
df3=df2.loc[:,'Adult Mortality':'Schooling']
intercept=np.ones([len(df3),1])
df4=np.c_[intercept,df3]
b=df2.iloc[:,:1]

# LU decomposition
x=np.array(df4)
from scipy.linalg import lu
xtx=np.dot(x.T,x) # x'x
p,l,u=lu(xtx) # Ax=b=plu

xy=np.dot(x.T,b) # x'y
pxy=np.dot(p.T,xy) # p'x'y
bs=np.linalg.solve(l,pxy) # Lb=p'x'y
np.linalg.solve(u,bs) # ux=b

%%time
# Cholesky decomposition
from scipy.linalg import cho_factor, cho_solve
l,lh = cho_factor(np.dot(x.T,x)) # L*L.H=A
cho_solve((l,lh),np.dot(x.T,b)) # A'Ax=A'b
