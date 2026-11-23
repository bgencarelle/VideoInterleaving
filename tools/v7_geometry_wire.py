"""Coherent sparse residuals on the current production V7 encoder.

Eight global Fourier atoms use coarse/fine frequency and quadrature amplitude.
Every adaptive parameter, including conventional/geometry selection, is carried
in existing ordinary luma slots. No tile joins, source cache or temporal state.
"""
import hashlib
from dataclasses import replace
import numpy as np
from numba import njit
from animation_modem import v7
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_color_metrics import resize_rgb, render as production_render
from aspect_fold import layout_size


@njit(cache=True)
def fit_atom(residual, fx, fy):
    """Variable projection: solve amplitudes, refine continuous frequency."""
    rows, cols = residual.shape
    a = b = 0.
    cx = np.empty(cols); sx = np.empty(cols)
    cy = np.empty(rows); sy = np.empty(rows)
    for iteration in range(6):
        for x in range(cols):
            angle = 2*np.pi*fx*((x+.5)/cols-.5)
            cx[x],sx[x] = np.cos(angle),np.sin(angle)
        for y in range(rows):
            angle = 2*np.pi*fy*((y+.5)/rows-.5)
            cy[y],sy[y] = np.cos(angle),np.sin(angle)
        cc = ss = cs = rc = rs = 0.
        for y in range(rows):
            for x in range(cols):
                c = cx[x]*cy[y]-sx[x]*sy[y]
                s = sx[x]*cy[y]+cx[x]*sy[y]
                cc += c*c; ss += s*s; cs += c*s
                rc += residual[y,x]*c; rs += residual[y,x]*s
        determinant = max(cc*ss-cs*cs, 1e-12)
        a = (rc*ss-rs*cs)/determinant
        b = (rs*cc-rc*cs)/determinant
        if iteration == 5:
            break
        xxsum = yysum = xysum = rx = ry = 0.
        for y in range(rows):
            yy = (y+.5)/rows-.5
            for x in range(cols):
                xx = (x+.5)/cols-.5
                c = cx[x]*cy[y]-sx[x]*sy[y]
                s = sx[x]*cy[y]+cx[x]*sy[y]
                derivative = 2*np.pi*(-a*s+b*c)
                dx, dy = xx*derivative, yy*derivative
                error = residual[y,x]-a*c-b*s
                xxsum += dx*dx; yysum += dy*dy; xysum += dx*dy
                rx += dx*error; ry += dy*error
        xxsum += 1e-9; yysum += 1e-9
        determinant = max(xxsum*yysum-xysum*xysum, 1e-15)
        delta_x=min(.2,max(-.2,(rx*yysum-ry*xysum)/determinant))
        delta_y=min(.2,max(-.2,(ry*xxsum-rx*xysum)/determinant))
        if max(abs(delta_x),abs(delta_y))<1e-5:
            break
        fx += delta_x
        fy += delta_y
    return fx, fy, a, b


@njit(cache=True)
def synthesize(atoms, width, height):
    output = np.zeros((height,width))
    for k in range(len(atoms)):
        fx,fy,a,b = atoms[k]
        if a == 0. and b == 0.:
            continue
        # Separate trigonometric factors avoid one sin/cos per output pixel.
        cx = np.empty(width); sx = np.empty(width)
        for x in range(width):
            angle = 2*np.pi*fx*((x+.5)/width-.5)
            cx[x],sx[x] = np.cos(angle),np.sin(angle)
        for y in range(height):
            angle = 2*np.pi*fy*((y+.5)/height-.5)
            cy,sy = np.cos(angle),np.sin(angle)
            for x in range(width):
                output[y,x] += a*(cx[x]*cy-sx[x]*sy)+b*(sx[x]*cy+cx[x]*sy)
    return output


class GeometryWire(ProductionFrameWire):
    ATOMS = 8
    PARAMETERS = 10
    FREQUENCY_LIMIT = 96.
    AMPLITUDE_LIMIT = .5
    MIN_GAIN = .10

    def __init__(self,profile,layout,enabled=True):
        super().__init__(profile,layout)
        self.enabled = enabled
        sent = np.unique(np.concatenate(self.model.rank_tables))
        self.sent = sent[sent>=0]
        always_sent=self.sent.copy()
        for rank in self.model.rank_tables:
            always_sent=np.intersect1d(always_sent,rank[rank>=0])
        candidates = np.setdiff1d(always_sent,self.codec.hosts)
        positions = self.codec.kept[candidates]
        candidates = candidates[(positions>0)&(positions<self.codec.grid.off[1])]
        # Source variance measures the cost of displacing a picture coefficient,
        # not carrier precision. Use low-cost ordinary slots, but give each
        # parameter the same *transmitted* gain-domain range. The old sd scaling
        # coupled precision to picture variance and unnecessarily removed the
        # strongest detail coefficients.
        transmitted_ranges=4*self.model.gain[candidates]*np.sqrt(self.model.lam[candidates])
        self.transmitted_parameter_range=float(np.quantile(transmitted_ranges,.75))
        order = np.argsort(self.model.lam[candidates],kind='stable')
        # Put marker redundancy in separate OFDM blocks, so it is not three
        # copies exposed to the same block erasure.
        block_for={int(rank):int(v7.GROUP_BLOCK[group])
                   for group,row in enumerate(self.model.rank_tables[0])
                   for rank in row if rank>=0}
        markers=[];used_blocks=set()
        for index in candidates[order]:
            block=block_for[int(index)]
            if block not in used_blocks:
                markers.append(index);used_blocks.add(block)
                if len(markers)==3:break
        if len(markers)!=3:raise ValueError('insufficient independent marker blocks')
        markers=np.asarray(markers)
        candidates=candidates[~np.isin(candidates,markers)]
        order = np.argsort(self.model.lam[candidates],kind='stable')
        # Repeat coarse coordinates across distinct blocks. Fine offsets and
        # amplitudes remain single analog coordinates; every copy costs a slot.
        available=list(candidates[order]);indices=[]
        for atom in range(self.ATOMS):
            for copies in self.coordinate_groups():
                blocks=set()
                for copy in range(copies):
                    index=next(i for i in available if block_for[int(i)] not in blocks)
                    available.remove(index);indices.append(index);blocks.add(block_for[int(index)])
        self.indices = np.asarray(indices)
        self.positions = self.codec.kept[self.indices]
        if len(self.indices)!=self.ATOMS*self.PARAMETERS:
            raise ValueError('insufficient always-carried ordinary parameter slots')
        self.mean = self.model.mu[self.indices]
        self.scale = self.transmitted_parameter_range/self.model.gain[self.indices]
        self.marker_indices=markers
        self.marker_positions=self.codec.kept[markers]
        self.marker_means=self.model.mu[markers]
        self.marker_scales=self.transmitted_parameter_range/self.model.gain[markers]
        self.parameter_scale = np.array([self.FREQUENCY_LIMIT,self.FREQUENCY_LIMIT,
                                         self.AMPLITUDE_LIMIT,self.AMPLITUDE_LIMIT])
        self.wire_parameter_scale = np.array([self.FREQUENCY_LIMIT]*6+[.5,.5,1.,1.])
        self.last_diagnostics = {}
        if enabled:
            self._install_parameter_priors()

    def _install_parameter_priors(self):
        """The MMSE receiver must use the variance of the carried parameters.

        This changes statistical decoding priors only. It does not change the
        rank tables, modulation gain, fold statistics or emitted phase/scale.
        The same preconfigured rule is used by independently created receivers.
        """
        original=self.model
        lam=original.lam.copy()
        parameter_variance=self.carrier_variances()
        if parameter_variance.shape!=(len(self.indices),):
            raise ValueError('parameter prior count does not match carried coefficients')
        lam[self.indices]=np.maximum(lam[self.indices],self.scale**2*parameter_variance)
        lam[self.marker_indices]=np.maximum(lam[self.marker_indices],self.marker_scales**2)
        priors=tuple(v7.block_priors(original.gain,lam,rank) for rank in original.rank_tables)
        model=replace(original,lam=lam,lam32=np.asarray(lam,np.float32),
                      block_prior_tables=priors,
                      block_prior_tables32=tuple(np.asarray(p,np.float32) for p in priors))
        if hasattr(original,'_mono_wire_profile'):
            model._mono_wire_profile=original._mono_wire_profile
        for key,value in self.wire._models.items():
            if value is original:self.wire._models[key]=model
        if self.profile=='aspect-fold-500':
            self.wire._codecs[id(model)]=self.codec
        else:
            self.wire._codec_by_model_id[id(model)]=self.codec
        self.model=model

    def coordinate_variances(self):
        return [1/3]*8+[.5,.5]

    def carrier_variances(self):
        return np.tile(self.coordinate_variances(),self.ATOMS)

    def coordinate_groups(self):
        return [3,3]+[1]*(self.PARAMETERS-6)

    def parameters(self,coefficients):
        normalized = np.clip((coefficients[self.positions]-self.mean)/self.scale,-1,1)
        coordinates=normalized.reshape(self.ATOMS,self.PARAMETERS)*self.wire_parameter_scale
        atoms=np.empty((self.ATOMS,4))
        atoms[:,0]=np.round(np.median(coordinates[:,:3],axis=1))+coordinates[:,6]
        atoms[:,1]=np.round(np.median(coordinates[:,3:6],axis=1))+coordinates[:,7]
        amplitudes=coordinates[:,8:]
        atoms[:,2:]=self.AMPLITUDE_LIMIT*amplitudes*abs(amplitudes)
        return atoms

    def with_parameters(self,coefficients,parameters):
        parameters=np.asarray(parameters)
        coordinates=np.empty((self.ATOMS,self.PARAMETERS))
        coarse=np.round(parameters[:,:2])
        coordinates[:,:3]=coarse[:,0:1]
        coordinates[:,3:6]=coarse[:,1:2]
        coordinates[:,6:8]=parameters[:,:2]-coarse
        amplitudes=parameters[:,2:]/self.AMPLITUDE_LIMIT
        coordinates[:,8:]=np.sign(amplitudes)*np.sqrt(abs(amplitudes))
        normalized = coordinates/self.wire_parameter_scale
        if np.max(abs(normalized))>1+1e-12:
            raise ValueError('geometry parameters exceed the explicit wire domain')
        result = coefficients.copy()
        result[self.positions] = self.mean+self.scale*normalized.ravel()
        result[self.marker_positions] = self.marker_means+self.marker_scales
        return result

    def base_values(self,coefficients,geometry=True):
        base = coefficients.copy()
        if geometry:
            base[self.positions] = self.mean
        base[self.marker_positions] = self.marker_means
        return self.codec.grid.inverse(base)

    def clean_projection(self,coefficients):
        """Production fold's deterministic noiseless support/compander model.

        This is an encoder-side model, not scored channel decoding. Actual audio
        encoding/decoding remains mandatory in the benchmark.
        """
        codec = self.codec
        full = np.zeros_like(coefficients)
        full[codec.kept] = self.model.mu
        full[codec.kept[self.sent]] = coefficients[codec.kept[self.sent]]
        host = (coefficients[codec.kept[codec.hosts]]-self.model.mu[codec.hosts])/codec.sd_host
        full[codec.kept[codec.hosts]] = self.model.mu[codec.hosts]+codec.sd_host*codec.D*np.round(host/codec.D)
        full[codec.guests] = codec.sd_guest*codec._expand(np.clip(codec._compress(coefficients[codec.guests]/codec.sd_guest),-1,1))
        full[codec.kept[codec.hosts[-codec.signature:]]] = self.model.mu[codec.hosts[-codec.signature:]]
        full[codec.guests[-codec.signature:]] = 0.
        return full

    def fit_residual(self,residual):
        remaining = np.ascontiguousarray(residual)
        atoms = np.zeros((self.ATOMS,4))
        rows,cols = remaining.shape
        for k in range(self.ATOMS):
            spectrum = abs(np.fft.rfft2(remaining))**2
            spectrum[0,0] = 0.
            fy_grid = np.fft.fftfreq(rows)*rows
            fx_grid = np.arange(spectrum.shape[1])
            spectrum[(abs(fy_grid[:,None])>self.FREQUENCY_LIMIT)|(fx_grid[None,:]>self.FREQUENCY_LIMIT)] = 0.
            iy,ix = np.unravel_index(np.argmax(spectrum),spectrum.shape)
            fx,fy,a,b = fit_atom(remaining,float(ix),float(fy_grid[iy]))
            if abs(fx)>self.FREQUENCY_LIMIT or abs(fy)>self.FREQUENCY_LIMIT:
                break
            if max(abs(a),abs(b))>self.AMPLITUDE_LIMIT or a*a+b*b<1e-8:
                break
            atoms[k] = fx,fy,a,b
            remaining = remaining-synthesize(atoms[k:k+1],cols,rows)
        return atoms

    def values(self,rgb):
        original = ProductionFrameWire.values(self,rgb)
        coefficients = self.codec.grid.forward(original)
        if not self.enabled:
            return self.codec.grid.inverse(coefficients)
        w,h = layout_size(self.layout)
        size = (round(192*w/min(w,h)),round(192*h/min(w,h)))
        projected = self.clean_projection(coefficients)
        conventional = production_render(self,self.base_values(projected,False),size,False)
        base = production_render(self,self.base_values(projected),size,False)
        # Float32 is the input format Pillow's F-mode resizer uses anyway;
        # avoid a full native-resolution float64 allocation and three casts.
        source = resize_rgb(np.asarray(rgb,np.float32)*np.float32(1/255),size)
        # Equal RGB addition is a luma-only correction. Its least-squares
        # target is the RGB-channel mean error, not mismatched color weights.
        residual = np.mean(source-base,axis=-1)
        atoms = self.fit_residual(residual)
        correction = self.correction(atoms,size)
        old_error = float(np.mean((source-conventional)**2))
        new_error = float(np.mean((source-base-correction[...,None])**2))
        selected = new_error<old_error*(1-self.MIN_GAIN)
        self.last_diagnostics = {'geometry_selected':bool(selected),
                                 'predicted_conventional_mse':old_error,
                                 'predicted_geometry_mse':new_error,
                                 'parameter_slots':len(self.positions)+len(self.marker_positions)}
        if selected:
            result = self.with_parameters(coefficients,atoms)
        else:
            result = coefficients.copy()
            result[self.marker_positions] = self.marker_means-self.marker_scales
        return self.codec.grid.inverse(result)

    @staticmethod
    def correction(parameters,size):
        return synthesize(np.ascontiguousarray(parameters),int(size[0]),int(size[1]))

    def geometry_selected(self,coefficients):
        markers=(coefficients[self.marker_positions]-self.marker_means)/self.marker_scales
        return bool(np.median(markers)>0)

    def render_received(self,values,size,display=False):
        if not self.enabled:
            return production_render(self,values,size,display)
        coefficients = self.codec.grid.forward(values)
        selected = self.geometry_selected(coefficients)
        base = self.base_values(coefficients,selected)
        rgb = production_render(self,base,size,display)
        if selected:
            rgb = rgb+self.correction(self.parameters(coefficients),size)[...,None]
        return rgb

    def record(self):
        record = super().record()
        digest = hashlib.sha256(self.positions.tobytes()+self.scale.tobytes()).hexdigest()
        record.update(source_model='eight coherent Fourier residual atoms with in-band conventional fallback',
                      atoms=self.ATOMS,parameters_per_atom=self.PARAMETERS,
                      parameter_positions=self.positions.tolist(),parameter_scale=self.parameter_scale.tolist(),
                      parameter_carrier_scale=self.scale.tolist(),marker_positions=self.marker_positions.tolist(),
                      transmitted_parameter_range=self.transmitted_parameter_range,
                      receiver_priors='uniform frequency and square-root amplitude variances; bipolar marker variance; production fold statistics retained',
                      carrier_selection='least source variance; equal transmitted gain-domain range',
                      frequency_coordinates='rounded whole-image cycle count plus analog subcycle offset',
                      wire_parameter_scale=self.wire_parameter_scale.tolist(),
                      amplitude_mapping='signed square root; exact signed square inverse',
                      coarse_frequency_redundancy='three copies in independent OFDM blocks; median decision',
                      geometry_digest=digest,source_fitting='current frame only; Numba variable projection',
                      minimum_predicted_mse_gain=self.MIN_GAIN,enabled=self.enabled,
                      headers_unchanged=True,component_ratios_unchanged=True,source_cache=False)
        return record
