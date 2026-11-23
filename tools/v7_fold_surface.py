"""Dimension-reducing serpentine analog mappings on fixed fold guest slots."""
import copy
import hashlib
import numpy as np
from tools.v7_analog_frame import ProductionFrameWire
from test_modem_v7.aspect_fold import layout_size

DIMENSIONS = ((9, 1), (7, 2), (5, 2), (7, 3), (3, 2), (1, 1))


def surface_encode(x, outputs, strips):
    """Keep the first outputs coordinates; interleave quantized extra axes.

    Each added axis folds an existing coordinate into alternating strips.
    Source clipping/strip quantization are explicit lossy operations.
    """
    source = np.clip(np.asarray(x), 0., np.nextafter(1., 0.))
    result = source[..., :outputs].copy()
    for i in range(outputs, source.shape[-1]):
        axis = (i-outputs) % outputs
        layer = np.floor(source[..., i]*strips).astype(np.int64)
        inner = np.where(layer % 2 == 0, result[..., axis], 1-result[..., axis])
        inner = np.clip(inner,0.,np.nextafter(1.,0.))
        result[..., axis] = (layer+inner)/strips
    return result


def surface_decode(x, inputs, strips):
    outputs = x.shape[-1]
    working = np.clip(np.asarray(x), 0., np.nextafter(1., 0.)).copy()
    result = np.empty((*working.shape[:-1], inputs))
    for i in range(inputs-1, outputs-1, -1):
        axis = (i-outputs) % outputs
        scaled = working[..., axis]*strips
        layer = np.minimum(np.floor(scaled).astype(np.int64), strips-1)
        fraction = scaled-layer
        result[..., i] = (layer+.5)/strips
        working[..., axis] = np.where(layer % 2 == 0, fraction, 1-fraction)
    result[..., :outputs] = working
    return result


class FoldSurfaceWire(ProductionFrameWire):
    def __init__(self, profile, layout, inputs, outputs, strips, training, normalization='fold-native', headroom=1.):
        if not 1 <= outputs <= inputs or strips < 2:
            raise ValueError('require inputs >= outputs >= 1 and strips >= 2')
        super().__init__(profile, layout)
        self.inputs, self.outputs, self.strips = inputs, outputs, strips
        if not np.isfinite(headroom) or not 0 < headroom <= 1:
            raise ValueError('headroom must be in (0,1]')
        self.headroom = float(headroom)
        if normalization not in ('linear','arctan','identity','fold-native'):
            raise ValueError(normalization)
        if normalization=='identity' and (inputs!=outputs or headroom!=1):
            raise ValueError('identity control requires equal dimensions and full amplitude')
        self.normalization = normalization
        old = self.codec
        self.codec = copy.copy(old)
        codec = self.codec
        usable = codec.M-codec.signature
        self.groups = usable//outputs
        self.carriers = codec.guests[:self.groups*outputs].copy()
        # Every group retains its old physical guests as the first coordinates;
        # extra coordinates are actual previously uncarried luma frequencies.
        source_positions = np.empty((self.groups, inputs), int)
        source_positions[:, :outputs] = self.carriers.reshape(self.groups, outputs)
        sent = np.unique(np.concatenate(self.model.rank_tables))
        sent = sent[sent >= 0]
        occupied = np.union1d(codec.kept[sent], codec.guests)
        occupied = np.union1d(occupied, codec.kept[codec.hosts])
        extra = np.setdiff1d(np.arange(codec.grid.off[1]), occupied)
        cols = codec.grid.grids[0][1]
        u,v = extra//cols, extra % cols
        width,height = layout_size(layout)
        extra = extra[np.argsort(u*u*width*width+v*v*height*height, kind='stable')]
        count = self.groups*(inputs-outputs)
        if count > len(extra):
            raise ValueError(f'needs {count} extra coordinates; fixed source grid has {len(extra)}')
        source_positions[:, outputs:] = extra[:count].reshape(self.groups, inputs-outputs)
        self.source_positions = source_positions
        self.source_compander = copy.copy(old)
        self.source_sd=np.empty(source_positions.shape)
        self.source_sd[:,:outputs]=old.sd_guest[:self.groups*outputs].reshape(self.groups,outputs)
        if normalization=='identity':
            # A true no-op through the experimental wrapper: preserve the
            # production fold scale/signature, source values and decoded values.
            self.mean=np.zeros(source_positions.shape)
            self.range=np.ones(source_positions.shape)
            self.amplitude=old.sd_guest[:self.groups*outputs].reshape(self.groups,outputs).copy()
            self.mapping_digest='production-identity-control'
            return
        coeffs = np.stack([codec.grid.forward(ProductionFrameWire.values(self,rgb)) for rgb in training])
        selected = coeffs[:, source_positions]
        if inputs>outputs:
            self.source_sd[:,outputs:]=np.maximum(np.sqrt(np.mean(selected[:,:,outputs:]**2,axis=0)),.01)
        self.mean = selected.mean(0)
        # Three-sigma ranges with a declared original-scale floor. Every source
        # overload is reported; no validation pixels fit this mapping.
        floor = np.broadcast_to(old.sd_guest[:self.groups*outputs].reshape(self.groups,outputs).mean(1)[:,None],self.mean.shape)*.1
        self.range = np.maximum(3*selected.std(0,ddof=1),np.maximum(floor,.01))
        encoded = surface_encode(self.normalize(selected),outputs,strips)*2-1
        self.amplitude = old.sd_guest[:self.groups*outputs].reshape(self.groups,outputs).copy()
        codec.sd_guest = old.sd_guest.copy()
        # Calibrate at full amplitude and keep this scale fixed across the
        # headroom sweep. Refitting it after attenuation cancels the experiment.
        if normalization=='fold-native':
            # Source coordinates already inhabit the production fold's
            # compressed domain. Carry them directly instead of companding
            # them a second time. Preserve the original physical guest scale.
            codec._compress=lambda u: np.clip(u,-1.,1.)
            codec._expand=lambda u: np.asarray(u)
        else:
            codec.sd_guest[:self.groups*outputs] = np.maximum(np.sqrt(np.mean(encoded*encoded,axis=0)).ravel()*self.amplitude.ravel(),old.sd_guest[:self.groups*outputs]*.05)
        digest = hashlib.sha256()
        digest.update(f'{inputs}:{outputs}:{strips}:{normalization}'.encode())
        for array in (source_positions,self.mean,self.range,self.amplitude,self.source_sd):
            digest.update(array.tobytes())
        self.mapping_digest = digest.hexdigest()
        if not (normalization=='fold-native' and inputs==outputs):
            original_table = codec.table
            codec.table = lambda: dict(original_table(),surface_mapping_digest=self.mapping_digest,
                                       surface_linear_fold_domain=normalization=='fold-native')
            codec._set_identity()
        if profile == 'aspect-fold-500':
            self.wire._codecs[id(self.model)] = codec
        else:
            self.wire._codec_by_model_id[id(self.model)] = codec
            for key,value in list(self.wire._codecs.items()):
                if value is old:
                    self.wire._codecs[key] = codec

    def normalize(self,selected):
        if self.normalization=='fold-native':
            return self.source_compander._compress(selected/self.source_sd)/2+.5
        x = (selected-self.mean)/self.range
        return .5+np.arctan(x)/np.pi if self.normalization=='arctan' else x/2+.5

    def denormalize(self,normalized):
        if self.normalization=='fold-native':
            return self.source_compander._expand(np.clip(normalized*2-1,-1.,1.))*self.source_sd
        if self.normalization=='arctan':
            # Retain explicit finite precision at endpoints. Bound the inverse
            # to the physical DCT coefficient range implied by source values
            # in [-1,1], rather than inventing unbounded detail on a bad decode.
            inner = np.clip(normalized,1e-12,1-1e-12)
            result = self.mean+self.range*np.tan(np.pi*(inner-.5))
            physical_bound = np.sqrt(self.codec.grid.off[1])
            return np.clip(result,-physical_bound,physical_bound)
        return self.mean+(normalized-.5)*2*self.range

    def source_diagnostics(self,rgb):
        if self.normalization=='identity':
            return {'source_range_overload_fraction':0.,'mapping_source_only_coefficient_mse':0.,
                    'extra_source_coordinates':0,'coordinate_ratio':1.,'carrier_headroom':1.,
                    'maximum_strip_depth':0,'inverse_amplitude_gain_db':0.}
        coefficients = self.codec.grid.forward(ProductionFrameWire.values(self,rgb))
        selected = coefficients[self.source_positions]
        normalized = self.normalize(selected)
        reconstructed = surface_decode(surface_encode(normalized,self.outputs,self.strips),self.inputs,self.strips)
        error = self.denormalize(reconstructed)-selected
        packed = (surface_encode(normalized,self.outputs,self.strips)*2-1)*self.headroom
        return {'source_range_overload_fraction':float(np.mean((normalized<0)|(normalized>=1))),
                'carrier_headroom':self.headroom,
                'packed_carrier_abs_peak':float(np.max(abs(packed))),
                'packed_carrier_endpoint_fraction':float(np.mean(abs(packed)>.95)),
                'inverse_amplitude_gain_db':float(-20*np.log10(self.headroom)),
                'outside_training_linear_range_fraction':float(np.mean(abs(selected-self.mean)>self.range)),
                'mapping_source_only_coefficient_mse':float(np.mean(error*error)),
                'extra_source_coordinates':int(self.groups*(self.inputs-self.outputs)),
                'coordinate_ratio':self.inputs/self.outputs,
                'maximum_strip_depth':int(np.ceil((self.inputs-self.outputs)/self.outputs))}

    def values(self,rgb):
        if self.normalization=='identity':
            return ProductionFrameWire.values(self,rgb)
        coefficients = self.codec.grid.forward(ProductionFrameWire.values(self,rgb))
        normalized = self.normalize(coefficients[self.source_positions])
        mapped = surface_encode(normalized,self.outputs,self.strips)*2-1
        coefficients[self.carriers] = (mapped*self.amplitude*self.headroom).ravel()
        return self.codec.grid.inverse(coefficients)

    def decode(self,audio,rate=96000):
        if self.normalization=='identity':
            return super().decode(audio,rate)
        import time
        started = time.perf_counter()
        packets,info,_ = super().decode(audio,rate)
        pictures=[]
        for result,values in packets:
            if values is None:
                pictures.append((result,None))
                continue
            coefficients = self.codec.grid.forward(values)
            normalized = coefficients[self.carriers].reshape(self.groups,self.outputs)/(self.amplitude*self.headroom)/2+.5
            source = surface_decode(normalized,self.inputs,self.strips)
            coefficients[self.source_positions.ravel()] = self.denormalize(source).ravel()
            pictures.append((result,self.codec.grid.inverse(coefficients)))
        return pictures,info,time.perf_counter()-started

    def record(self):
        record=super().record()
        record.update(mapping='serpentine folded surface',dimensions=[self.inputs,self.outputs],strips=self.strips,
                      mapping_digest=self.mapping_digest,source_positions=self.source_positions.tolist(),
                      mean=self.mean.tolist(),range=self.range.tolist(),amplitude=self.amplitude.tolist(),
                      fixed_carrier_geometry=True,extra_source_coordinates=self.groups*(self.inputs-self.outputs),
                      carrier_headroom=self.headroom,headroom_calibration='full-amplitude scales and signature held fixed',
                      normalization=self.normalization,range_fit='training mean +/- three sigma with floor',
                      source_sd=self.source_sd.tolist(),
                      meaning='source coordinate ratio, not measured spatial resolution')
        return record
