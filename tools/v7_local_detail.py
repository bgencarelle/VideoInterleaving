"""Experimental overlapping Gabor residuals in existing V7 picture slots.

No runtime decode-error gate: partially received parameters are rendered as-is.
The clean-wire qualification threshold belongs solely to the benchmark.
"""
import numpy as np
from numba import njit
from tools.v7_geometry_wire import GeometryWire
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_color_metrics import resize_rgb,render as production_render
from aspect_fold import layout_size


@njit(cache=True)
def scaled_carriers(scale,multiplier):
    return scale*multiplier


@njit(cache=True)
def source_units(rgb):
    return rgb.astype(np.float32)/np.float32(255)


@njit(cache=True)
def residual_values(source,base):
    rows,cols,_=source.shape;residual=np.empty((rows,cols))
    for y in range(rows):
        for x in range(cols):
            residual[y,x]=0.
            for c in range(3):residual[y,x]+=(source[y,x,c]-base[y,x,c])/3
    return residual


@njit(cache=True)
def predicted_errors(source,conventional,base,correction):
    old=new=0.
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            for c in range(3):
                old+=(source[y,x,c]-conventional[y,x,c])**2
                new+=(source[y,x,c]-base[y,x,c]-correction[y,x])**2
    return old/source.size,new/source.size


@njit(cache=True)
def conventional_branch(coefficients,markers,means,scales):
    result=coefficients.copy()
    for i in range(len(markers)):result[markers[i]]=means[i]-scales[i]
    return result


@njit(cache=True)
def local_synthesize(atoms, width, height):
    output=np.zeros((height,width))
    for atom in atoms:
        fx,fy,a,b,cx,cy,wx,wy=atom
        if a==0 and b==0:continue
        cosx=np.empty(width);sinx=np.empty(width);ex=np.empty(width)
        for x in range(width):
            xx=(x+.5)/width-.5
            cosx[x]=np.cos(2*np.pi*fx*xx);sinx[x]=np.sin(2*np.pi*fx*xx)
            ex[x]=np.exp(-.5*((xx-cx)/wx)**2)
        for y in range(height):
            yy=(y+.5)/height-.5
            ey=np.exp(-.5*((yy-cy)/wy)**2)
            cosy=np.cos(2*np.pi*fy*yy);siny=np.sin(2*np.pi*fy*yy)
            for x in range(width):
                output[y,x]+=ex[x]*ey*(a*(cosx[x]*cosy-sinx[x]*siny)+b*(sinx[x]*cosy+cosx[x]*siny))
    return output


@njit(cache=True)
def fft_inplace(values):
    """Radix-two FFT used inside the compiled source search."""
    n=len(values);j=0
    for i in range(1,n):
        bit=n>>1
        while j&bit:
            j^=bit;bit>>=1
        j^=bit
        if i<j:values[i],values[j]=values[j],values[i]
    length=2
    while length<=n:
        step=np.exp(-2j*np.pi/length)
        for start in range(0,n,length):
            weight=1.+0j
            for k in range(length//2):
                u=values[start+k];v=values[start+k+length//2]*weight
                values[start+k]=u+v;values[start+k+length//2]=u-v
                weight*=step
        length*=2


@njit(cache=True)
def spectral_peak(residual,window,limit):
    rows,cols=residual.shape
    nr=1;nc=1
    while nr<rows:nr*=2
    while nc<cols:nc*=2
    spectrum=np.zeros((nr,nc),np.complex128)
    spectrum[:rows,:cols]=residual*window
    for y in range(nr):fft_inplace(spectrum[y])
    for x in range(nc):
        column=spectrum[:,x].copy();fft_inplace(column);spectrum[:,x]=column
    best=0.;fx=fy=0.
    for y in range(nr):
        frequency_y=(y if y<=nr//2 else y-nr)*rows/nr
        if abs(frequency_y)>limit:continue
        for x in range(nc//2+1):
            frequency_x=x*cols/nc
            if frequency_x>limit or (x==0 and y==0):continue
            value=abs(spectrum[y,x])**2
            if value>best:best=value;fx=frequency_x;fy=frequency_y
    return fx,fy


@njit(cache=True)
def dictionary_windows(rows,cols):
    windows=np.empty((32,rows,cols));centers=np.empty((32,3));index=0
    for width in (.12,.24):
        for cy in (-.375,-.125,.125,.375):
            for cx in (-.375,-.125,.125,.375):
                centers[index]=cx,cy,width
                for y in range(rows):
                    yy=(y+.5)/rows-.5
                    for x in range(cols):
                        xx=(x+.5)/cols-.5
                        windows[index,y,x]=np.exp(-.5*(((xx-cx)/width)**2+((yy-cy)/width)**2))
                index+=1
    return windows,centers


@njit(cache=True)
def fit_local(residual,windows,centers,count,limit,amplitude_limit):
    remaining=residual.copy();rows,cols=residual.shape
    atoms=np.zeros((count,8));atoms[:,6:]=.16
    for k in range(count):
        energies=np.zeros(len(windows))
        for w in range(len(windows)):
            weight=0.
            for y in range(rows):
                for x in range(cols):
                    squared=windows[w,y,x]**2
                    energies[w]+=remaining[y,x]**2*squared;weight+=squared
            energies[w]/=weight
        best_gain=0.;best=np.zeros(8)
        for candidate in np.argsort(energies)[-4:]:
            cx,cy,width=centers[candidate]
            fx,fy=spectral_peak(remaining,windows[candidate],limit)
            atom=np.array([[fx,fy,1.,0.,cx,cy,width,width]])
            c=local_synthesize(atom,cols,rows)
            atom[0,2]=0;atom[0,3]=1
            s=local_synthesize(atom,cols,rows)
            cc=np.sum(c*c)+1e-12;ss=np.sum(s*s)+1e-12;cs=np.sum(c*s)
            rc=np.sum(remaining*c);rs=np.sum(remaining*s)
            determinant=max(cc*ss-cs*cs,1e-15)
            a=(rc*ss-rs*cs)/determinant;b=(rs*cc-rc*cs)/determinant
            if max(abs(a),abs(b))>amplitude_limit:continue
            gain=2*(a*rc+b*rs)-(a*a*cc+2*a*b*cs+b*b*ss)
            if gain>best_gain:best_gain=gain;best=np.array([fx,fy,a,b,cx,cy,width,width])
        if best_gain<1e-8:break
        atoms[k]=best
        remaining-=local_synthesize(atoms[k:k+1],cols,rows)
    # Joint box-constrained least squares using a small Gram system. Coordinate
    # descent includes cross-atom terms and never emits out-of-domain strengths.
    basis=np.zeros((count*2,rows*cols))
    for k in range(count):
        if atoms[k,2]==0 and atoms[k,3]==0:continue
        for component in range(2):
            unit=atoms[k:k+1].copy();unit[0,2:4]=0;unit[0,2+component]=1
            basis[2*k+component]=local_synthesize(unit,cols,rows).ravel()
    gram=np.zeros((count*2,count*2));target=np.zeros(count*2)
    flat_residual=residual.ravel()
    for i in range(count*2):
        for pixel in range(rows*cols):target[i]+=basis[i,pixel]*flat_residual[pixel]
        for j in range(i+1):
            for pixel in range(rows*cols):gram[i,j]+=basis[i,pixel]*basis[j,pixel]
            gram[j,i]=gram[i,j]
    amplitudes=atoms[:,2:4].copy().ravel()
    for iteration in range(80):
        change=0.
        for j in range(len(amplitudes)):
            if gram[j,j]<1e-12:continue
            other=0.
            for i in range(len(amplitudes)):
                if i!=j:other+=gram[j,i]*amplitudes[i]
            value=(target[j]-other)/gram[j,j]
            value=min(amplitude_limit,max(-amplitude_limit,value))
            change=max(change,abs(value-amplitudes[j]));amplitudes[j]=value
        if change<1e-7:break
    atoms[:,2:4]=amplitudes.reshape(count,2)
    return atoms


@njit(cache=True)
def encode_atoms(coefficients,atoms,positions,mean,scale,markers,marker_mean,marker_scale):
    if atoms.shape[1]!=8 or len(atoms)*32!=len(positions):raise ValueError('invalid local detail shape')
    result=coefficients.copy()
    for k in range(len(atoms)):
        fx,fy,a,b,cx,cy,wx,wy=atoms[k]
        if abs(fx)>96 or abs(fy)>96 or abs(a)>.5 or abs(b)>.5 or abs(cx)>.5 or abs(cy)>.5 or wx<.06 or wx>.32 or wy<.06 or wy>.32:
            raise ValueError('local detail atoms exceed explicit wire domain')
        coords=np.empty(32)
        coords[:3]=np.round(fx)/96;coords[3:6]=np.round(fy)/96
        coords[6]=(fx-np.round(fx))/.5;coords[7]=(fy-np.round(fy))/.5
        amp_a=np.sign(a)*np.sqrt(abs(a)/.5);amp_b=np.sign(b)*np.sqrt(abs(b)/.5)
        ca=np.round(amp_a*64);cb=np.round(amp_b*64)
        coords[8:11]=ca/64;coords[11:14]=cb/64
        coords[14]=(amp_a*64-ca)*2;coords[15]=(amp_b*64-cb)*2
        coarse_x=np.round(cx*128);coarse_y=np.round(cy*128)
        coords[16:19]=coarse_x/64;coords[19:22]=coarse_y/64
        coords[22]=(cx*128-coarse_x)*2;coords[23]=(cy*128-coarse_y)*2
        width_x=np.round((wx-.19)*256);width_y=np.round((wy-.19)*256)
        coords[24:27]=width_x/64;coords[27:30]=width_y/64
        coords[30]=((wx-.19)*256-width_x)*2;coords[31]=((wy-.19)*256-width_y)*2
        for j in range(32):
            if not np.isfinite(coords[j]) or abs(coords[j])>1+1e-12:raise ValueError('local detail atoms exceed explicit wire domain')
            index=k*32+j;result[positions[index]]=mean[index]+scale[index]*coords[j]
    for i in range(len(markers)):result[markers[i]]=marker_mean[i]+marker_scale[i]
    return result


@njit(cache=True)
def decode_atoms(coefficients,positions,mean,scale):
    atoms=np.empty((len(positions)//32,8))
    for k in range(len(atoms)):
        coords=np.empty(32)
        for j in range(32):
            i=k*32+j;coords[j]=min(1.,max(-1.,(coefficients[positions[i]]-mean[i])/scale[i]))
        x=np.sort(coords[:3]);y=np.sort(coords[3:6])
        atoms[k,0]=min(96.,max(-96.,np.round(x[1]*96)+coords[6]*.5))
        atoms[k,1]=min(96.,max(-96.,np.round(y[1]*96)+coords[7]*.5))
        a=(np.round(np.sort(coords[8:11])[1]*64)+coords[14]*.5)/64
        b=(np.round(np.sort(coords[11:14])[1]*64)+coords[15]*.5)/64
        atoms[k,2]=.5*a*abs(a);atoms[k,3]=.5*b*abs(b)
        cx=np.sort(coords[16:19]);cy=np.sort(coords[19:22])
        atoms[k,4]=(np.round(cx[1]*64)+coords[22]*.5)/128
        atoms[k,5]=(np.round(cy[1]*64)+coords[23]*.5)/128
        atoms[k,6]=min(.32,max(.06,.19+(np.round(np.sort(coords[24:27])[1]*64)+coords[30]*.5)/256))
        atoms[k,7]=min(.32,max(.06,.19+(np.round(np.sort(coords[27:30])[1]*64)+coords[31]*.5)/256))
    return atoms


class LocalDetailWire(GeometryWire):
    """Eight smooth directional atoms; 256 coordinates + three markers.

    Physical atom: fx, fy, cosine/sine amplitude, center x/y, width x/y.
    Widths are normalized image units, not pixels. Each frame stands alone.
    """
    PARAMETERS=32
    WIDTH_MIN=.06
    WIDTH_MAX=.32

    def __init__(self,profile,layout,enabled=True,refine=True):
        super().__init__(profile,layout,False)
        self.enabled=enabled
        self.refine_geometry=refine
        # Local position/width coordinates need tighter absolute recovery than
        # the global-wave codec. Charge and measure the increased carrier range.
        self.scale=scaled_carriers(self.scale,2.)
        self.parameter_scale=np.array([96.,96.,.5,.5,.5,.5,.26,.26])
        self.wire_parameter_scale=np.array([96.]*6+[.5,.5]+[64.]*6+[.5,.5]+[64.]*6+[.5,.5]+[64.]*6+[.5,.5])
        self._window_cache={}
        if enabled:self._install_parameter_priors()

    def coordinate_variances(self):
        return [1/3]*32

    def carrier_variances(self):
        return np.full(len(self.indices),1/3)

    def coordinate_groups(self):
        return [3,3,1,1]*4

    def with_parameters(self,coefficients,parameters):
        return encode_atoms(coefficients,np.asarray(parameters,float),self.positions,self.mean,self.scale,
                            self.marker_positions,self.marker_means,self.marker_scales)

    def parameters(self,coefficients):
        return decode_atoms(coefficients,self.positions,self.mean,self.scale)

    @staticmethod
    def correction(parameters,size):
        return local_synthesize(np.ascontiguousarray(parameters),int(size[0]),int(size[1]))

    def fit_residual(self,residual):
        rows,cols=residual.shape
        cache_key=(rows,cols)
        if cache_key not in self._window_cache:
            self._window_cache[cache_key]=dictionary_windows(rows,cols)
        windows,centers=self._window_cache[cache_key]
        atoms=fit_local(residual,windows,centers,self.ATOMS,self.FREQUENCY_LIMIT,self.AMPLITUDE_LIMIT)
        if self.refine_geometry:
            from tools.v7_local_detail_fit import refine_local
            atoms=refine_local(residual,atoms)
        self.fitted_atoms=atoms
        return atoms

    def values(self,rgb):
        original=ProductionFrameWire.values(self,rgb)
        coefficients=self.codec.grid.forward(original)
        if not self.enabled:return self.codec.grid.inverse(coefficients)
        w,h=layout_size(self.layout)
        size=(round(192*w/min(w,h)),round(192*h/min(w,h)))
        projected=self.clean_projection(coefficients)
        conventional=production_render(self,self.base_values(projected,False),size,False)
        base=production_render(self,self.base_values(projected),size,False)
        source=resize_rgb(source_units(rgb),size)
        atoms=self.fit_residual(residual_values(source,base))
        correction=self.correction(atoms,size)
        old_error,new_error=predicted_errors(source,conventional,base,correction)
        selected=new_error<old_error*(1-self.MIN_GAIN)
        self.last_diagnostics={'geometry_selected':bool(selected),'predicted_conventional_mse':old_error,
                               'predicted_geometry_mse':new_error,'parameter_slots':len(self.positions)+3}
        result=(self.with_parameters(coefficients,atoms) if selected else
                conventional_branch(coefficients,self.marker_positions,self.marker_means,self.marker_scales))
        return self.codec.grid.inverse(result)

    def force_values(self,rgb):
        adaptive=LocalDetailWire.values(self,rgb)
        coefficients=self.codec.grid.forward(adaptive)
        return self.codec.grid.inverse(self.with_parameters(coefficients,self.fitted_atoms))

    def record(self):
        record=super().record()
        record.update(source_model='overlapping Gaussian directional residual atoms',
                      physical_parameters=['fx','fy','cosine','sine','center_x','center_y','width_x','width_y'],
                      parameter_slots=len(self.positions)+3,width_domain=[self.WIDTH_MIN,self.WIDTH_MAX],
                      source_fitting='current-frame spectral seed and continuous pixel+gradient joint geometry refinement' if self.refine_geometry else 'historical discrete geometry and joint amplitude refit',
                      refinement_enabled=self.refine_geometry,
                      parameter_transmitted_range_multiplier=2,
                      receiver_priors='uniform coarse/fine analog-coordinate variances; production fold statistics retained',
                      local_coordinate_mapping='triple coarse and single fine offset per physical parameter; signed-square-root amplitudes',
                      dictionary_cache_only=True,
                      runtime_error_rejection=False)
        return record
