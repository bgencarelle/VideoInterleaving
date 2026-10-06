"""Fixed orthonormal high-band residual modes; one analog value per mode."""
import numpy as np
from numba import njit
from tools.v7_geometry_wire import GeometryWire
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_color_metrics import resize_rgb,render as production_render
from aspect_fold import layout_size
from tools.v7_local_detail import source_units,residual_values


@njit(cache=True)
def fixed_modes(count,height,width,low=24.,high=92.):
    if count%16:raise ValueError('mode count must be divisible by 16')
    modes=np.empty((count,2),np.int64);bins=count//16;found=0
    for radial in range(bins):
        radius=low+(radial+.5)*(high-low)/bins
        for direction in range(16):
            angle=(direction+.5)*(np.pi/2)/16
            vertical=int(np.round(2*radius*np.sin(angle)))
            horizontal=int(np.round(2*radius*np.cos(angle)))
            if vertical>=height or horizontal>=width:continue
            duplicate=False
            for j in range(found):
                if modes[j,0]==vertical and modes[j,1]==horizontal:
                    duplicate=True;break
            if not duplicate and found<count:
                modes[found,0]=vertical;modes[found,1]=horizontal;found+=1
    if found!=count:raise ValueError('fixed mode lattice contains duplicate/out-of-range frequencies')
    return modes


@njit(cache=True)
def dct_factors(modes,height,width):
    count=len(modes);vertical=np.empty((count,height));horizontal=np.empty((count,width))
    for k in range(count):
        u,v=modes[k]
        au=np.sqrt(1/height) if u==0 else np.sqrt(2/height)
        av=np.sqrt(1/width) if v==0 else np.sqrt(2/width)
        for y in range(height):vertical[k,y]=au*np.cos(np.pi*(y+.5)*u/height)
        for x in range(width):horizontal[k,x]=av*np.cos(np.pi*(x+.5)*v/width)
    return vertical,horizontal


@njit(cache=True)
def project_modes(residual,vertical,horizontal):
    output=np.zeros(len(vertical));rows,cols=residual.shape
    for k in range(len(output)):
        for y in range(rows):
            for x in range(cols):output[k]+=residual[y,x]*vertical[k,y]*horizontal[k,x]
    return output


@njit(cache=True)
def synthesize_modes(coefficients,vertical,horizontal):
    rows=len(vertical[0]);cols=len(horizontal[0]);output=np.zeros((rows,cols))
    for k in range(len(coefficients)):
        for y in range(rows):
            for x in range(cols):output[y,x]+=coefficients[k]*vertical[k,y]*horizontal[k,x]
    return output


@njit(cache=True)
def encode_mode_values(values,base,positions,means,scales,markers,marker_means,marker_scales,limit):
    result=base.copy();saturated=0
    for i in range(len(values)):
        normalized=values[i]/limit
        if abs(normalized)>1:saturated+=1;normalized=np.sign(normalized)
        normalized=np.sign(normalized)*np.sqrt(abs(normalized))
        result[positions[i]]=means[i]+scales[i]*normalized
    for i in range(len(markers)):result[markers[i]]=marker_means[i]+marker_scales[i]
    return result,saturated


@njit(cache=True)
def decode_mode_values(coefficients,positions,means,scales,limit):
    output=np.empty(len(positions))
    for i in range(len(output)):
        z=min(1.,max(-1.,(coefficients[positions[i]]-means[i])/scales[i]))
        output[i]=limit*z*abs(z)
    return output


@njit(cache=True)
def detail_mse(source,conventional,base,detail):
    old=new=0.
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):
            for c in range(3):
                old+=(source[y,x,c]-conventional[y,x,c])**2
                new+=(source[y,x,c]-base[y,x,c]-detail[y,x])**2
    return old/source.size,new/source.size


@njit(cache=True)
def fallback_values(coefficients,markers,means,scales):
    result=coefficients.copy()
    for i in range(len(markers)):result[markers[i]]=means[i]-scales[i]
    return result


@njit(cache=True)
def add_luma(rgb,detail):
    result=rgb.copy()
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            for c in range(3):result[y,x,c]+=detail[y,x]
    return result


class ResidualDetailWire(GeometryWire):
    """K fixed high-band cosine residual values carried in K+3 old slots."""
    ATOMS=32
    PARAMETERS=8
    COUNT=256
    COEFFICIENT_LIMIT=8.
    LOW_CYCLES=24.
    HIGH_CYCLES=92.

    def __init__(self,profile,layout,count=None,enabled=True):
        super().__init__(profile,layout,False)
        self.count=int(self.COUNT if count is None else count)
        if self.count<16 or self.count>len(self.indices) or self.count%16:
            raise ValueError('residual coefficient count must be 16..256, divisible by 16')
        self.enabled=enabled
        self.indices=self.indices[:self.count].copy()
        self.positions=self.codec.kept[self.indices]
        self.mean=self.model.mu[self.indices]
        self.coefficient_scale=np.full(self.count,self.COEFFICIENT_LIMIT)
        self.scale=self.coefficient_scale/self.model.gain[self.indices]
        self.parameter_scale=self.coefficient_scale.copy()
        self.modes={};self.factors={};self.last_diagnostics={}
        if enabled:self._install_parameter_priors()

    def carrier_variances(self):
        return np.full(len(self.indices),1/3)

    def basis(self,size):
        key=tuple(size)
        if key not in self.modes:
            width,height=key
            modes=fixed_modes(self.count,height,width,self.LOW_CYCLES,self.HIGH_CYCLES)
            self.modes[key]=modes
            self.factors[key]=dct_factors(modes,height,width)
        return self.modes[key],self.factors[key]

    def parameters(self,coefficients):
        return decode_mode_values(coefficients,self.positions,self.mean,self.scale,self.COEFFICIENT_LIMIT)

    def with_parameters(self,coefficients,parameters):
        result,saturated=encode_mode_values(np.asarray(parameters,float),coefficients,
            self.positions,self.mean,self.scale,self.marker_positions,self.marker_means,
            self.marker_scales,self.COEFFICIENT_LIMIT)
        self.last_diagnostics['saturated_coefficients']=saturated
        return result

    def correction(self,parameters,size):
        _,(vertical,horizontal)=self.basis(size)
        return synthesize_modes(np.asarray(parameters,float),vertical,horizontal)

    def values(self,rgb):
        original=ProductionFrameWire.values(self,rgb)
        coefficients=self.codec.grid.forward(original)
        if not self.enabled:return self.codec.grid.inverse(coefficients)
        width,height=layout_size(self.layout)
        size=(round(192*width/min(width,height)),round(192*height/min(width,height)))
        projected=self.clean_projection(coefficients)
        conventional=production_render(self,self.base_values(projected,False),size,False)
        base=production_render(self,self.base_values(projected),size,False)
        source=resize_rgb(source_units(rgb),size)
        residual=residual_values(source,base)
        _,factors=self.basis(size)
        modes=project_modes(residual,*factors)
        result=self.with_parameters(coefficients,modes)
        carried=self.parameters(result)
        detail=synthesize_modes(carried,*factors)
        self.fitted_modes=carried.copy()
        old_error,new_error=detail_mse(source,conventional,base,detail)
        selected=new_error<old_error*(1-self.MIN_GAIN)
        output=result if selected else fallback_values(coefficients,self.marker_positions,self.marker_means,self.marker_scales)
        self.last_diagnostics={'geometry_selected':bool(selected),'predicted_conventional_mse':old_error,
            'predicted_geometry_mse':new_error,'saturated_coefficients':self.last_diagnostics.get('saturated_coefficients',0),
            'parameter_slots':self.count+3,'mode_count':self.count}
        return self.codec.grid.inverse(output)

    def render_received(self,values,size,display=False):
        coefficients=self.codec.grid.forward(values)
        selected=self.geometry_selected(coefficients)
        base=self.base_values(coefficients,selected)
        rgb=production_render(self,base,size,display)
        if selected:rgb=add_luma(rgb,self.correction(self.parameters(coefficients),size))
        return rgb

    def record(self):
        record=super().record()
        record.update(source_model='fixed orthonormal high-band DCT residual coefficients',
            independent_detail_values=self.count,parameter_slots=self.count+3,
            fixed_frequency_band_cycles=[self.LOW_CYCLES,self.HIGH_CYCLES],
            coefficient_mapping='signed square-root analog amplitudes with explicit saturation count',
            adaptive_mode_indices=False,entropy_bitstream=False,digital_quantization=False)
        return record
