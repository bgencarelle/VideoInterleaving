"""Continuous, fine-detail-weighted source refinement; native Numba only."""
import numpy as np
from numba import njit
from tools.v7_local_detail import local_synthesize


@njit(cache=True)
def weighted_inner(a,b,gradient_weight):
    result=0.
    rows,cols=a.shape
    for y in range(rows):
        for x in range(cols):
            result+=a[y,x]*b[y,x]
            if x+1<cols:result+=gradient_weight*(a[y,x+1]-a[y,x])*(b[y,x+1]-b[y,x])
            if y+1<rows:result+=gradient_weight*(a[y+1,x]-a[y,x])*(b[y+1,x]-b[y,x])
    return result


@njit(cache=True)
def amplitude_fit(target,atom,gradient_weight,amplitude_limit):
    rows,cols=target.shape
    unit=atom.reshape(1,8).copy();unit[0,2]=1;unit[0,3]=0
    c=local_synthesize(unit,cols,rows)
    unit[0,2]=0;unit[0,3]=1
    s=local_synthesize(unit,cols,rows)
    cc=weighted_inner(c,c,gradient_weight)+1e-12
    ss=weighted_inner(s,s,gradient_weight)+1e-12
    cs=weighted_inner(c,s,gradient_weight)
    rc=weighted_inner(target,c,gradient_weight);rs=weighted_inner(target,s,gradient_weight)
    determinant=max(cc*ss-cs*cs,1e-15)
    a=(rc*ss-rs*cs)/determinant;b=(rs*cc-rc*cs)/determinant
    # Exact box-constrained two-coordinate solve, including boundary solutions.
    best_a=0.;best_b=0.;best_cost=0.
    for case in range(5):
        if case==0:
            ca=a;cb=b
            if max(abs(ca),abs(cb))>amplitude_limit:continue
        elif case<3:
            ca=amplitude_limit if case==1 else -amplitude_limit
            cb=min(amplitude_limit,max(-amplitude_limit,(rs-cs*ca)/ss))
        else:
            cb=amplitude_limit if case==3 else -amplitude_limit
            ca=min(amplitude_limit,max(-amplitude_limit,(rc-cs*cb)/cc))
        cost=ca*ca*cc+2*ca*cb*cs+cb*cb*ss-2*(ca*rc+cb*rs)
        if cost<best_cost:best_cost=cost;best_a=ca;best_b=cb
    fitted=atom.copy();fitted[2]=best_a;fitted[3]=best_b
    return fitted,c,s


@njit(cache=True)
def solve_small(matrix,rhs):
    """Pivoted native solve; no NumPy/SciPy solver or BLAS dependency."""
    n=len(rhs);work=np.empty((n,n+1))
    work[:,:n]=matrix;work[:,n]=rhs
    for column in range(n):
        pivot=column
        for row in range(column+1,n):
            if abs(work[row,column])>abs(work[pivot,column]):pivot=row
        if abs(work[pivot,column])<1e-15:return np.zeros(n)
        if pivot!=column:
            for j in range(column,n+1):work[column,j],work[pivot,j]=work[pivot,j],work[column,j]
        for row in range(column+1,n):
            ratio=work[row,column]/work[column,column]
            for j in range(column,n+1):work[row,j]-=ratio*work[column,j]
    solution=np.empty(n)
    for row in range(n-1,-1,-1):
        value=work[row,n]
        for j in range(row+1,n):value-=work[row,j]*solution[j]
        solution[row]=value/work[row,row]
    return solution


@njit(cache=True)
def refine_atom(target,initial,gradient_weight,iterations):
    atom,c,s=amplitude_fit(target,initial,gradient_weight,.5)
    rows,cols=target.shape
    error=target-atom[2]*c-atom[3]*s
    cost=weighted_inner(error,error,gradient_weight)
    parameter_indices=np.array([0,1,4,5,6,7])
    step_limits=np.array([.5,.5,.06,.06,.04,.04])
    for iteration in range(iterations):
        jacobian=np.empty((6,rows,cols))
        a,b,cx,cy,wx,wy=atom[2:]
        for y in range(rows):
            yy=(y+.5)/rows-.5
            for x in range(cols):
                xx=(x+.5)/cols-.5
                wave=a*c[y,x]+b*s[y,x];phase=-a*s[y,x]+b*c[y,x]
                jacobian[0,y,x]=2*np.pi*xx*phase
                jacobian[1,y,x]=2*np.pi*yy*phase
                jacobian[2,y,x]=wave*(xx-cx)/(wx*wx)
                jacobian[3,y,x]=wave*(yy-cy)/(wy*wy)
                jacobian[4,y,x]=wave*(xx-cx)**2/(wx**3)
                jacobian[5,y,x]=wave*(yy-cy)**2/(wy**3)
        normal=np.empty((6,6));rhs=np.empty(6)
        for i in range(6):
            rhs[i]=weighted_inner(jacobian[i],error,gradient_weight)
            for j in range(i+1):
                value=weighted_inner(jacobian[i],jacobian[j],gradient_weight)
                normal[i,j]=value;normal[j,i]=value
        for i in range(6):normal[i,i]+=normal[i,i]*.003+1e-9
        step=solve_small(normal,rhs)
        ratio=1.
        for i in range(6):ratio=max(ratio,abs(step[i])/step_limits[i])
        step/=ratio
        accepted=False
        for scale in (1.,.5,.25,.125):
            candidate=atom.copy()
            for i in range(6):candidate[parameter_indices[i]]+=step[i]*scale
            if abs(candidate[0])>96 or abs(candidate[1])>96:continue
            if abs(candidate[4])>.5 or abs(candidate[5])>.5:continue
            if candidate[6]<.06 or candidate[6]>.32 or candidate[7]<.06 or candidate[7]>.32:continue
            candidate,nc,ns=amplitude_fit(target,candidate,gradient_weight,.5)
            ne=target-candidate[2]*nc-candidate[3]*ns
            new_cost=weighted_inner(ne,ne,gradient_weight)
            if new_cost<cost-1e-12:
                atom=candidate;c=nc;s=ns;error=ne;cost=new_cost;accepted=True
                break
        if not accepted:break
    return atom


@njit(cache=True)
def refine_local(residual,initial,gradient_weight=2.,sweeps=2,iterations=5):
    """Block-coordinate joint refinement: each atom fits the other atoms' residual.

    A move is accepted only if the shared pixel+gradient objective decreases.
    The atom count, physical domains and transmitted parameter format stay fixed.
    """
    atoms=initial.copy();rows,cols=residual.shape
    total=local_synthesize(atoms,cols,rows)
    for sweep in range(sweeps):
        for k in range(len(atoms)):
            if atoms[k,2]==0 and atoms[k,3]==0:continue
            old=local_synthesize(atoms[k:k+1],cols,rows)
            target=residual-total+old
            atom=refine_atom(target,atoms[k],gradient_weight,iterations)
            new=local_synthesize(atom.reshape(1,8),cols,rows)
            total+=new-old;atoms[k]=atom
    return atoms
