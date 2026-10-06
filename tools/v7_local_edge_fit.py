"""Source-only contour residual prototype. No edge wire mapping is installed."""
from math import erf
import numpy as np
from numba import njit
from tools.v7_local_detail_fit import weighted_inner


@njit(cache=True)
def edge_synthesize(atoms,width,height):
    """Sharp-minus-blurred edge, smoothly localized along its tangent.

    Physical coordinates: center x/y, normal angle, length, sharp/blurred widths,
    contrast, curvature. It supplies a detail correction, not a tile brightness.
    """
    output=np.zeros((height,width))
    for atom in atoms:
        cx,cy,angle,length,sharp,blurred,contrast,curvature=atom
        if contrast==0:continue
        ca=np.cos(angle);sa=np.sin(angle)
        for y in range(height):
            yy=(y+.5)/height-.5-cy
            for x in range(width):
                xx=(x+.5)/width-.5-cx
                v=-xx*sa+yy*ca;u=xx*ca+yy*sa+curvature*v*v
                window=np.exp(-.5*(v/length)**2)
                output[y,x]+=.5*contrast*window*(erf(u/sharp)-erf(u/blurred))
    return output


@njit(cache=True)
def fit_edges(residual,source,count,gradient_weight=2.):
    rows,cols=residual.shape;remaining=residual.copy()
    atoms=np.zeros((count,8));atoms[:,3]=.12;atoms[:,4]=.008;atoms[:,5]=.04
    for k in range(count):
        # Source gradient supplies orientation; the missing source detail
        # supplies priority. Nonmaximum suppression gives four distinct seeds.
        priority=np.zeros((rows,cols));gx=np.zeros_like(priority);gy=np.zeros_like(priority)
        for y in range(1,rows-1):
            for x in range(1,cols-1):
                gx[y,x]=(source[y,x+1]-source[y,x-1])*.5
                gy[y,x]=(source[y+1,x]-source[y-1,x])*.5
                gradient=gx[y,x]**2+gy[y,x]**2
                local=0.
                for dy in (-1,0,1):
                    for dx in (-1,0,1):local+=remaining[y+dy,x+dx]**2
                priority[y,x]=gradient*local
        best_gain=0.;best=atoms[k].copy()
        for seed in range(4):
            flat=np.argmax(priority);y=flat//cols;x=flat%cols
            if priority[y,x]<=1e-15:break
            angle=np.arctan2(gy[y,x]*rows,gx[y,x]*cols)
            cx=(x+.5)/cols-.5;cy=(y+.5)/rows-.5
            for shift in (-.15,0.,.15):
                for length in (.06,.12,.24):
                    for sharp in (.004,.012):
                        for blurred in (.025,.05):
                            normal=(angle+shift+np.pi)%(2*np.pi)-np.pi
                            atom=np.array([[cx,cy,normal,length,sharp,blurred,1.,0.]])
                            unit=edge_synthesize(atom,cols,rows)
                            energy=weighted_inner(unit,unit,gradient_weight)+1e-12
                            projection=weighted_inner(remaining,unit,gradient_weight)
                            strength=min(.5,max(-.5,projection/energy))
                            gain=2*strength*projection-strength*strength*energy
                            if gain>best_gain:
                                best_gain=gain;best=atom[0].copy();best[6]=strength
            radius=max(2,min(rows,cols)//12)
            for yy in range(max(0,y-radius),min(rows,y+radius+1)):
                for xx in range(max(0,x-radius),min(cols,x+radius+1)):priority[yy,xx]=0
        if best_gain<1e-9:break
        atoms[k]=best;remaining-=edge_synthesize(atoms[k:k+1],cols,rows)
    # Joint bounded strength refinement for intersecting/overlapping contours.
    basis=np.empty((count,rows,cols))
    for k in range(count):
        atom=atoms[k:k+1].copy();atom[0,6]=1
        basis[k]=edge_synthesize(atom,cols,rows)
    gram=np.empty((count,count));target=np.empty(count)
    for i in range(count):
        target[i]=weighted_inner(residual,basis[i],gradient_weight)
        for j in range(count):gram[i,j]=weighted_inner(basis[i],basis[j],gradient_weight)
    strengths=atoms[:,6].copy()
    for iteration in range(60):
        change=0.
        for i in range(count):
            other=0.
            for j in range(count):
                if j!=i:other+=gram[i,j]*strengths[j]
            value=min(.5,max(-.5,(target[i]-other)/max(gram[i,i],1e-12)))
            change=max(change,abs(value-strengths[i]));strengths[i]=value
        if change<1e-7:break
    atoms[:,6]=strengths
    return atoms
