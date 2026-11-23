"""Sparse source-selected DCT residual modes with analog coordinates and amplitudes.

Each mode costs seven old slots: three copies of each integer frequency and one
continuous signed-square-root amplitude. No entropy/digital bitstream or temporal
prediction. Frequencies, including source choices, arrive in that frame's audio.
"""
import numpy as np
from numba import njit
from tools.v7_geometry_wire import GeometryWire
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_color_metrics import render as production_render,resize_rgb
from tools.v7_local_detail import source_units,residual_values
from tools.v7_local_detail_metrics import fft
from tools.v7_residual_detail import detail_mse,fallback_values,add_luma
from aspect_fold import layout_size


@njit(cache=True)
def dct_vector(values):
    n=len(values);extended=np.empty(2*n,np.complex128)
    for i in range(n):extended[i]=values[i];extended[2*n-1-i]=values[i]
    spectrum=fft(extended);result=np.empty(n)
    for k in range(n):
        normalization=np.sqrt(1/n) if k==0 else np.sqrt(2/n)
        result[k]=.5*normalization*(spectrum[k]*np.exp(-1j*np.pi*k/(2*n))).real
    return result


@njit(cache=True)
def source_dct(residual):
    rows,cols=residual.shape;intermediate=np.empty_like(residual);result=np.empty_like(residual)
    for y in range(rows):intermediate[y]=dct_vector(residual[y])
    for x in range(cols):result[:,x]=dct_vector(intermediate[:,x].copy())
    return result


@njit(cache=True)
def choose_modes(residual,count):
    spectrum=source_dct(residual);rows,cols=spectrum.shape
    amplitudes=np.zeros((count,3))
    spectrum[0,0]=0
    for y in range(rows):
        for x in range(cols):
            if y>191 or x>191:spectrum[y,x]=0
    for k in range(count):
        best=0.;iy=ix=0
        for y in range(rows):
            for x in range(cols):
                if abs(spectrum[y,x])>best:best=abs(spectrum[y,x]);iy=y;ix=x
        if best<1e-12:break
        amplitudes[k,0]=iy;amplitudes[k,1]=ix
        amplitudes[k,2]=spectrum[iy,ix]/np.sqrt(rows*cols)
        spectrum[iy,ix]=0
    return amplitudes


@njit(cache=True)
def sparse_synthesize(modes,width,height):
    result=np.zeros((height,width))
    for k in range(len(modes)):
        u,v,a=modes[k]
        if a==0:continue
        ay=1. if u==0 else np.sqrt(2.)
        ax=1. if v==0 else np.sqrt(2.)
        cx=np.empty(width)
        for x in range(width):cx[x]=ax*np.cos(np.pi*(x+.5)*v/width)
        for y in range(height):
            cy=ay*np.cos(np.pi*(y+.5)*u/height)
            for x in range(width):result[y,x]+=a*cy*cx[x]
    return result


@njit(cache=True)
def insert_sparse(base,modes,positions,means,scales,markers,marker_means,marker_scales):
    output=base.copy()
    for k in range(len(modes)):
        u,v,a=modes[k]
        if u<0 or u>191 or v<0 or v>191 or abs(a)>.25:
            raise ValueError('sparse residual parameters exceed fixed domain')
        normalized=np.empty(7)
        normalized[:3]=u/191;normalized[3:6]=v/191
        normalized[6]=np.sign(a)*np.sqrt(abs(a)/.25)
        for j in range(7):
            i=k*7+j;output[positions[i]]=means[i]+scales[i]*normalized[j]
    for i in range(len(markers)):output[markers[i]]=marker_means[i]+marker_scales[i]
    return output


@njit(cache=True)
def read_sparse(coefficients,positions,means,scales):
    modes=np.empty((len(positions)//7,3))
    for k in range(len(modes)):
        coordinates=np.empty(7)
        for j in range(7):
            i=k*7+j;coordinates[j]=min(1.,max(-1.,(coefficients[positions[i]]-means[i])/scales[i]))
        u=np.sort(coordinates[:3]);v=np.sort(coordinates[3:6])
        modes[k,0]=max(0.,np.round(u[1]*191));modes[k,1]=max(0.,np.round(v[1]*191))
        modes[k,2]=.25*coordinates[6]*abs(coordinates[6])
    return modes


class SparseResidualWire(GeometryWire):
    ATOMS=8
    PARAMETERS=7
    MIN_GAIN=.01

    def __init__(self,profile,layout,enabled=True):
        super().__init__(profile,layout,enabled)
        self.parameter_scale=np.array([191.,191.,.25])

    def coordinate_groups(self):return [3,3,1]

    def carrier_variances(self):return np.full(len(self.indices),.5)

    def parameters(self,coefficients):
        return read_sparse(coefficients,self.positions,self.mean,self.scale)

    def with_parameters(self,coefficients,parameters):
        return insert_sparse(coefficients,np.asarray(parameters,float),self.positions,self.mean,self.scale,
            self.marker_positions,self.marker_means,self.marker_scales)

    @staticmethod
    def correction(parameters,size):return sparse_synthesize(np.asarray(parameters,float),int(size[0]),int(size[1]))

    def values(self,rgb):
        original=ProductionFrameWire.values(self,rgb);coefficients=self.codec.grid.forward(original)
        if not self.enabled:return self.codec.grid.inverse(coefficients)
        width,height=layout_size(self.layout)
        size=(round(192*width/min(width,height)),round(192*height/min(width,height)))
        projected=self.clean_projection(coefficients)
        conventional=production_render(self,self.base_values(projected,False),size,False)
        base=production_render(self,self.base_values(projected),size,False)
        source=resize_rgb(source_units(rgb),size)
        modes=choose_modes(residual_values(source,base),self.ATOMS)
        correction=self.correction(modes,size)
        old_error,new_error=detail_mse(source,conventional,base,correction)
        selected=new_error<old_error*(1-self.MIN_GAIN)
        self.fitted_modes=modes
        self.last_diagnostics={'geometry_selected':bool(selected),'predicted_conventional_mse':old_error,
            'predicted_geometry_mse':new_error,'parameter_slots':len(self.positions)+3}
        output=self.with_parameters(coefficients,modes) if selected else fallback_values(coefficients,self.marker_positions,self.marker_means,self.marker_scales)
        return self.codec.grid.inverse(output)

    def record(self):
        record=super().record()
        record.update(source_model='source-selected orthogonal DCT residual modes',
            physical_parameters=['vertical DCT index','horizontal DCT index','normalized modal amplitude'],
            mode_count=self.ATOMS,parameter_slots=len(self.positions)+3,
            entropy_bitstream=False,digital_compression=False,frame_independent=True,
            source_dct='native Numba mixed-radix DCT-II on the residual; production source uses current direct DCT')
        return record
