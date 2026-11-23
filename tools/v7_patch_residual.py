"""Coherent fixed-support residual waves with locally referenced phase.

Only four analog values per support: two frequencies and two strengths. Positions,
widths and phase origins are fixed receiver definitions, not per-frame overhead.
"""
import numpy as np
from numba import njit
from tools.v7_geometry_wire import GeometryWire
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_color_metrics import render as production_render,resize_rgb
from tools.v7_local_detail import source_units,residual_values,spectral_peak
from tools.v7_local_detail_fit import weighted_inner,solve_small
from tools.v7_residual_detail import detail_mse,fallback_values
from aspect_fold import layout_size


@njit(cache=True)
def supports(count):
    nx=int(np.ceil(np.sqrt(count)));ny=count//nx
    if nx*ny!=count:raise ValueError('patch count must form a rectangular lattice')
    output=np.empty((count,4));k=0
    for y in range(ny):
        for x in range(nx):
            output[k,0]=(x+.5)/nx-.5;output[k,1]=(y+.5)/ny-.5
            output[k,2]=.65/nx;output[k,3]=.65/ny;k+=1
    return output


@njit(cache=True)
def patch_unit(fx,fy,support,width,height,quadrature):
    cx,cy,wx,wy=support
    result=np.empty((height,width));xxcos=np.empty(width);xxsin=np.empty(width);ex=np.empty(width)
    for x in range(width):
        xx=(x+.5)/width-.5-cx
        xxcos[x]=np.cos(2*np.pi*fx*xx);xxsin[x]=np.sin(2*np.pi*fx*xx)
        ex[x]=np.exp(-.5*(xx/wx)**2)
    mean=0.
    for y in range(height):
        yy=(y+.5)/height-.5-cy
        cyy=np.cos(2*np.pi*fy*yy);syy=np.sin(2*np.pi*fy*yy)
        ey=np.exp(-.5*(yy/wy)**2)
        for x in range(width):
            wave=xxsin[x]*cyy+xxcos[x]*syy if quadrature else xxcos[x]*cyy-xxsin[x]*syy
            result[y,x]=ex[x]*ey*wave;mean+=result[y,x]
    # No independent patch brightness offsets; the combined correction has zero DC.
    mean/=height*width
    for y in range(height):
        for x in range(width):result[y,x]-=mean
    return result


@njit(cache=True)
def patch_synthesize(modes,definitions,width,height):
    result=np.zeros((height,width))
    for k in range(len(modes)):
        fx,fy,a,b=modes[k]
        if a==0 and b==0:continue
        c=patch_unit(fx,fy,definitions[k],width,height,False)
        s=patch_unit(fx,fy,definitions[k],width,height,True)
        result+=a*c+b*s
    return result


@njit(cache=True)
def fit_patches(residual,definitions):
    rows,cols=residual.shape;count=len(definitions);modes=np.zeros((count,4))
    basis=np.empty((count*2,rows,cols))
    remaining=residual.copy()
    for k in range(count):
        cx,cy,wx,wy=definitions[k];window=np.empty((rows,cols))
        for y in range(rows):
            for x in range(cols):
                xx=(x+.5)/cols-.5;yy=(y+.5)/rows-.5
                window[y,x]=np.exp(-.5*(((xx-cx)/wx)**2+((yy-cy)/wy)**2))
        fx,fy=spectral_peak(remaining,window,96.)
        modes[k,0]=fx;modes[k,1]=fy
        basis[2*k]=patch_unit(fx,fy,definitions[k],cols,rows,False)
        basis[2*k+1]=patch_unit(fx,fy,definitions[k],cols,rows,True)
        cc=weighted_inner(basis[2*k],basis[2*k],0.)
        ss=weighted_inner(basis[2*k+1],basis[2*k+1],0.)
        cs=weighted_inner(basis[2*k],basis[2*k+1],0.)
        rc=weighted_inner(remaining,basis[2*k],0.)
        rs=weighted_inner(remaining,basis[2*k+1],0.)
        determinant=max(cc*ss-cs*cs,1e-12)
        a=(rc*ss-rs*cs)/determinant;b=(rs*cc-rc*cs)/determinant
        modes[k,2]=min(.5,max(-.5,a));modes[k,3]=min(.5,max(-.5,b))
        remaining-=modes[k,2]*basis[2*k]+modes[k,3]*basis[2*k+1]
    gram=np.empty((count*2,count*2));target=np.empty(count*2)
    for i in range(count*2):
        target[i]=weighted_inner(residual,basis[i],0.)
        for j in range(count*2):gram[i,j]=weighted_inner(basis[i],basis[j],0.)
    coefficients=modes[:,2:4].copy().reshape(count*2)
    for iteration in range(100):
        change=0.
        for i in range(count*2):
            other=0.
            for j in range(count*2):
                if i!=j:other+=gram[i,j]*coefficients[j]
            value=min(.5,max(-.5,(target[i]-other)/max(gram[i,i],1e-12)))
            change=max(change,abs(value-coefficients[i]));coefficients[i]=value
        if change<1e-7:break
    modes[:,2:4]=coefficients.reshape(count,2)
    return refine_patches(residual,modes,definitions)


@njit(cache=True)
def refine_patches(residual,modes,definitions):
    """Bounded variable projection with fixed support and local phase origin."""
    rows,cols=residual.shape
    modes=modes.copy()
    reconstruction=patch_synthesize(modes,definitions,cols,rows)
    for sweep in range(3):
        for k in range(len(modes)):
            old=patch_synthesize(modes[k:k+1],definitions[k:k+1],cols,rows)
            target=residual-reconstruction+old
            current=modes[k].copy()
            for iteration in range(12):
                previous=current.copy()
                c=patch_unit(current[0],current[1],definitions[k],cols,rows,False)
                s=patch_unit(current[0],current[1],definitions[k],cols,rows,True)
                cc=weighted_inner(c,c,0.);ss=weighted_inner(s,s,0.);cs=weighted_inner(c,s,0.)
                rc=weighted_inner(target,c,0.);rs=weighted_inner(target,s,0.)
                determinant=max(cc*ss-cs*cs,1e-12)
                current[2]=min(.5,max(-.5,(rc*ss-rs*cs)/determinant))
                current[3]=min(.5,max(-.5,(rs*cc-rc*cs)/determinant))
                previous_error=target-previous[2]*c-previous[3]*s
                fitted_error=target-current[2]*c-current[3]*s
                if weighted_inner(fitted_error,fitted_error,0.)>weighted_inner(previous_error,previous_error,0.):
                    current[2:4]=previous[2:4]
                model=current[2]*c+current[3]*s;error=target-model
                derivatives=np.empty((2,rows,cols))
                for axis in range(2):
                    plus=current.copy();minus=current.copy()
                    plus[axis]+=.001;minus[axis]-=.001
                    derivatives[axis]=(patch_synthesize(plus.reshape(1,4),definitions[k:k+1],cols,rows)-
                        patch_synthesize(minus.reshape(1,4),definitions[k:k+1],cols,rows))/.002
                matrix=np.empty((2,2));rhs=np.empty(2)
                for i in range(2):
                    rhs[i]=weighted_inner(error,derivatives[i],0.)
                    for j in range(2):matrix[i,j]=weighted_inner(derivatives[i],derivatives[j],0.)
                    matrix[i,i]+=1e-9
                delta=solve_small(matrix,rhs)
                move=min(1.,.5/max(max(abs(delta[0]),abs(delta[1])),1e-12))
                accepted=False
                for trial in range(6):
                    candidate=current.copy()
                    for axis in range(2):candidate[axis]=min(96.,max(-96.,current[axis]+move*delta[axis]))
                    predicted=patch_synthesize(candidate.reshape(1,4),definitions[k:k+1],cols,rows)
                    if weighted_inner(target-predicted,target-predicted,0.)<weighted_inner(error,error,0.):
                        current=candidate;accepted=True;break
                    move*=.5
                if not accepted:break
            modes[k]=current
            reconstruction+=patch_synthesize(current.reshape(1,4),definitions[k:k+1],cols,rows)-old
    return modes


@njit(cache=True)
def patch_insert(base,modes,positions,means,scales,markers,marker_means,marker_scales):
    output=base.copy()
    for k in range(len(modes)):
        values=np.empty(4)
        values[0]=modes[k,0]/96;values[1]=modes[k,1]/96
        values[2]=np.sign(modes[k,2])*np.sqrt(abs(modes[k,2])/.5)
        values[3]=np.sign(modes[k,3])*np.sqrt(abs(modes[k,3])/.5)
        for j in range(4):
            if not np.isfinite(values[j]) or abs(values[j])>1+1e-12:raise ValueError('patch domain exceeded')
            i=k*4+j;output[positions[i]]=means[i]+scales[i]*values[j]
    for i in range(len(markers)):output[markers[i]]=marker_means[i]+marker_scales[i]
    return output


@njit(cache=True)
def patch_read(coefficients,positions,means,scales):
    modes=np.empty((len(positions)//4,4))
    for k in range(len(modes)):
        for j in range(4):
            i=k*4+j;value=min(1.,max(-1.,(coefficients[positions[i]]-means[i])/scales[i]))
            modes[k,j]=value*96 if j<2 else .5*value*abs(value)
    return modes


class PatchResidualWire(GeometryWire):
    ATOMS=16
    PARAMETERS=4
    MIN_GAIN=.01

    def __init__(self,profile,layout,enabled=True):
        super().__init__(profile,layout,enabled)
        self.definitions=supports(self.ATOMS)
        self.parameter_scale=np.array([96.,96.,.5,.5])

    def coordinate_groups(self):return [1,1,1,1]
    def carrier_variances(self):return np.tile(np.array([1/3,1/3,.5,.5]),self.ATOMS)
    def parameters(self,coefficients):return patch_read(coefficients,self.positions,self.mean,self.scale)
    def with_parameters(self,coefficients,parameters):
        return patch_insert(coefficients,np.asarray(parameters,float),self.positions,self.mean,self.scale,
            self.marker_positions,self.marker_means,self.marker_scales)
    def correction(self,parameters,size):return patch_synthesize(np.asarray(parameters,float),self.definitions,int(size[0]),int(size[1]))

    def values(self,rgb):
        original=ProductionFrameWire.values(self,rgb);coefficients=self.codec.grid.forward(original)
        if not self.enabled:return self.codec.grid.inverse(coefficients)
        width,height=layout_size(self.layout);size=(round(192*width/min(width,height)),round(192*height/min(width,height)))
        projected=self.clean_projection(coefficients)
        conventional=production_render(self,self.base_values(projected,False),size,False)
        base=production_render(self,self.base_values(projected),size,False)
        source=resize_rgb(source_units(rgb),size)
        modes=fit_patches(residual_values(source,base),self.definitions)
        detail=self.correction(modes,size)
        old_error,new_error=detail_mse(source,conventional,base,detail)
        selected=new_error<old_error*(1-self.MIN_GAIN)
        self.fitted_modes=modes
        self.last_diagnostics={'geometry_selected':bool(selected),'predicted_conventional_mse':old_error,
            'predicted_geometry_mse':new_error,'parameter_slots':len(self.positions)+3}
        output=self.with_parameters(coefficients,modes) if selected else fallback_values(coefficients,self.marker_positions,self.marker_means,self.marker_scales)
        return self.codec.grid.inverse(output)

    def record(self):
        record=super().record();record.update(source_model='fixed overlapping supports, local-phase analog residual waves',
            physical_parameters=['fx','fy','cosine amplitude','sine amplitude'],
            definitions=self.definitions.tolist(),parameter_slots=self.ATOMS*4+3,
            source_dct='current production direct DCT',per_frame_geometry_overhead=False,entropy_bitstream=False)
        return record
