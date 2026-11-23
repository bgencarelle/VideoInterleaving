"""Independent references for scope kernel migration and state continuity."""
import numpy as np
import pytest
import ast
from pathlib import Path

from scope_numeric import (area_grid, importance_grid, circular_filter,
                           positive_percentile, warm_scope_numeric)
from scope_bake import StochasticEmitter
import scope_numeric
import scope_out
from numba.core.registry import CPUDispatcher


@pytest.mark.parametrize('n', [31, 32, 1600, 3201])
@pytest.mark.parametrize('kind', [0, 1, 2])
def test_native_circular_transform_matches_numpy_at_actual_trace_lengths(n, kind):
    frame = np.random.default_rng(13).normal(size=(n, 2))
    frequency = np.fft.rfftfreq(n, 1/48000)
    if kind == 0:
        gain = 1/np.sqrt(1+(frequency/2300)**8)
    elif kind == 1:
        gain = .5-.5*np.cos(np.pi*np.clip((2300+1200-frequency)/1200, 0, 1))
    else:
        gain = np.ones(len(frequency), complex)
        gain[1:] = 1+120/(1j*frequency[1:])
        magnitude = abs(gain)
        gain[magnitude>8] *= 8/magnitude[magnitude>8]
        gain[0] = 0
    cutoff = 120 if kind == 2 else 2300
    expected = np.fft.irfft(np.fft.rfft(frame, axis=0)*gain[:, None], n=n, axis=0)
    actual = circular_filter(frame, 48000, cutoff, kind=kind, taper=1200, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_compiled_probability_preserves_sanitization_gradient_and_thresholds():
    image = np.random.default_rng(3).normal(.5, .4, size=(17, 23))
    image.flat[:3] = [np.nan, np.inf, -np.inf]
    clean = np.clip(np.nan_to_num(image, nan=0, posinf=1, neginf=0), 0, 1)
    gy, gx = np.gradient(clean)
    edge = np.hypot(gx, gy)
    edge /= edge.max()
    for stochastic in (False, True):
        expected = np.where(clean > (.2 if stochastic else .02), clean**2.2, 0)
        if stochastic:
            expected = np.clip(np.maximum(expected, .6*edge*(.25+.75*clean)), 0, 1)
        else:
            expected += .6*edge
        actual, _ = importance_grid(image, 2.2, .02, .6, stochastic)
        np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=3e-15)


def test_grid_area_and_percentile_match_independent_bin_references():
    image = np.random.default_rng(4).random((19, 27))
    expected = np.empty((7, 11))
    for y in range(7):
        for x in range(11):
            expected[y,x] = image[y*19//7:(y+1)*19//7, x*27//11:(x+1)*27//11].mean()
    np.testing.assert_allclose(area_grid(image, 7, 11), expected, atol=2e-14)
    for percentile in (2., 75., 98.):
        actual, count = positive_percentile(image, percentile, .3)
        assert count == np.count_nonzero(image>.3)
        assert actual == pytest.approx(np.percentile(image[image>.3], percentile), abs=1e-15)


def test_float32_area_grid_preserves_legacy_cumulative_rounding():
    image=np.random.default_rng(93).random((90,160),dtype=np.float32)
    integral=np.pad(image.cumsum(0).cumsum(1),((1,0),(1,0)))
    ys=np.arange(49)*90//48;xs=np.arange(81)*160//80
    expected=(integral[np.ix_(ys[1:],xs[1:])]-integral[np.ix_(ys[:-1],xs[1:])]
              -integral[np.ix_(ys[1:],xs[:-1])]+integral[np.ix_(ys[:-1],xs[:-1])])
    expected=expected/np.outer(np.diff(ys),np.diff(xs))
    np.testing.assert_array_equal(area_grid(image,48,80,True),expected)


@pytest.mark.parametrize('pattern',['constant','rising','falling','duplicates','random'])
def test_inplace_percentile_handles_large_and_degenerate_preview_fields(pattern):
    values=np.arange(20000,dtype=np.float64)
    if pattern=='constant':values[:]=1
    elif pattern=='falling':values=values[::-1].copy()
    elif pattern=='duplicates':values%=7
    elif pattern=='random':values=np.random.default_rng(5).random(len(values))
    for percentile in (0.,2.,50.,75.,98.,100.):
        actual,_=positive_percentile(values,percentile,-1.0)
        assert actual==pytest.approx(np.percentile(values,percentile),abs=1e-12)


def test_scope_has_no_interpreted_numpy_compute_or_scipy_imports():
    root=Path(__file__).resolve().parents[1]
    storage={
        'asarray','ascontiguousarray','array','empty','empty_like','zeros',
        'zeros_like','ones','ones_like','full','full_like','copyto','load','save',
        'savez','frombuffer','stack','vstack','hstack','concatenate','column_stack',
        'rot90','tile','repeat','int16','int32','int64','uint8','uint16','float32',
        'float64','bool_','ndarray','integer','dtype',
    }
    violations=[]

    class Audit(ast.NodeVisitor):
        def visit_FunctionDef(self,node):
            if any('njit' in ast.unparse(d) for d in node.decorator_list):return
            self.generic_visit(node)

        def visit_Call(self,node):
            function=node.func
            if isinstance(function,ast.Attribute):
                if isinstance(function.value,ast.Name) and function.value.id=='np':
                    if function.attr not in storage:
                        violations.append(f'{path.name}:{node.lineno}:np.{function.attr}')
                elif ast.unparse(function).startswith('np.fft.'):
                    violations.append(f'{path.name}:{node.lineno}:FFT')
            self.generic_visit(node)

        def visit_Import(self,node):
            if any(alias.name.startswith('scipy') for alias in node.names):
                violations.append(f'{path.name}:{node.lineno}:SciPy')

        def visit_ImportFrom(self,node):
            if (node.module or '').startswith('scipy'):
                violations.append(f'{path.name}:{node.lineno}:SciPy')

    for path in [*root.glob('scope*.py'),root/'tools/scope_screen.py',root/'utilities/convert_to_xy.py']:
        Audit().visit(ast.parse(path.read_text()))
    assert not violations, '\n'.join(violations)


@pytest.mark.parametrize('x_only',[False,True])
def test_output_mapping_and_guards_are_warmed_before_callback_use(x_only):
    scope=scope_out.Scope(device='null',samplerate=48000,samples=96,
                          channel_pair=(24,25),x_only=x_only,trigger=False)
    frame=np.random.default_rng(31).uniform(-.7,.7,(96,2)).astype(np.float32)
    readonly=frame.copy();readonly.setflags(write=False)
    kernels=[value for module in (scope_numeric,scope_out)
             for value in vars(module).values() if isinstance(value,CPUDispatcher)]
    before=[tuple(kernel.signatures) for kernel in kernels]
    for source in (frame,frame[::2],readonly,readonly[::2]):
        output=np.ones((len(source),scope.output_channels),np.float32)
        scope._write_output(output,source)
        assert not scope_out.beam_is_parked(source,x_only=x_only)
        scope.output_muted=True
        scope._write_output(output,source)
        scope.output_muted=False
        scope._write_output(output,source)
    after=[tuple(kernel.signatures) for kernel in kernels]
    assert before==after


def test_oversample_decimation_layout_is_warmed_before_trace_production():
    warm_scope_numeric()
    before=(tuple(scope_numeric.scale_float32.signatures),tuple(scope_numeric.clip_x.signatures))
    for dtype in (np.float32,np.float64):
        decimated=np.zeros((32,2),dtype)[::2]
        scope_numeric.scale_float32(decimated,.9)
    scope_numeric.clip_x(np.zeros((32,2),np.float64)[::2],-.9,.9)
    assert before==(tuple(scope_numeric.scale_float32.signatures),tuple(scope_numeric.clip_x.signatures))


@pytest.mark.parametrize('shape',[(1,31),(19,1),(19,31)])
def test_bake_candidates_preserve_quantized_coordinates_signal_edge_and_mass(shape):
    from scope_numeric import baked_stipple_candidates
    rng=np.random.default_rng(73)
    lum=rng.integers(0,256,shape,dtype=np.uint8)
    alpha=rng.integers(0,256,shape,dtype=np.uint8)
    light=lum.astype(float)/255
    coverage=alpha.astype(float)/255
    if min(shape)>1:
        gy,gx=np.gradient(light);edge=np.hypot(gx,gy)
        if edge.max()>1e-12:edge/=edge.max()
    else:edge=np.zeros(shape)
    proposal=coverage*(.15+.75*light+.10*edge)
    total=proposal.sum();count=128
    marks=(np.arange(count)+.5)*total/count
    chosen=np.clip(np.searchsorted(np.cumsum(proposal.ravel()),marks,side='left'),0,proposal.size-1)
    y,x=np.divmod(chosen,shape[1])
    expected_xy=np.column_stack((np.round(x*65535./max(shape[1]-1,1)),
                                 np.round(y*65535./max(shape[0]-1,1)))).astype(np.uint16)
    expected_lae=np.column_stack((lum[y,x],alpha[y,x],np.round(edge[y,x]*255))).astype(np.uint8)
    xy,lae,mass=baked_stipple_candidates(lum,alpha,count)
    np.testing.assert_array_equal(xy,expected_xy)
    np.testing.assert_array_equal(lae,expected_lae)
    assert mass==np.float32(total/proposal.size)


def test_bake_mask_thresholds_matte_and_geometry_quantization_match_reference():
    from scope_numeric import matte_mean,masked_quantiles,normalized_points,quantized_vertices
    rng=np.random.default_rng(83)
    rgb=rng.integers(0,256,(17,23,3),dtype=np.uint8)
    np.testing.assert_array_equal(matte_mean(rgb),rgb.mean(axis=2).astype(np.uint8))
    image=rgb[:,:,0];mask=rgb[:,:,1]
    expected=np.percentile(image[mask>0],np.linspace(0,100,6)[1:-1])
    np.testing.assert_array_equal(masked_quantiles(image,mask,4),expected)
    points=rng.uniform(-2,25,(31,2))
    expected=(points-np.array([23,17])/2)/(23/2)
    np.testing.assert_array_equal(normalized_points(points,23,17),expected)
    np.testing.assert_array_equal(quantized_vertices(expected),np.clip(np.round(expected*32767),-32767,32767).astype(np.int16))


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('samplerate', [48000, 96000])
def test_compiled_stochastic_walk_preserves_reference_rng_state_across_chunks(sparse, samplerate):
    warm_scope_numeric()
    probability = np.random.default_rng(71).uniform(.1, .9, (17, 23))
    if sparse:
        probability[:] = 0
        probability[3, 14] = .8
        probability[9, 4] = .5
    reference = StochasticEmitter(samplerate, 257, seed=19, radius=2)
    actual = StochasticEmitter(samplerate, 257, seed=19, radius=2)
    reference._ensure_state(probability.shape)
    step = min(reference.walk_hz, float(samplerate))/samplerate
    h,w = probability.shape
    m = float(max(h,w))
    for _ in range(3):
        expected = np.empty((257,2),np.float32)
        for i in range(257):
            if reference._phase >= 1:
                reference._advance(probability)
                reference._phase -= 1
            x,y = reference._pixel
            expected[i,0] = reference.level*(2*(x+(m-w)*.5)/m-1)
            expected[i,1] = reference.level*(1-2*(y+(m-h)*.5)/m)
            reference._phase += step
        received = actual.emit_probability(probability)
        np.testing.assert_array_equal(received, expected)
        np.testing.assert_array_equal(actual._visited, reference._visited)
        assert actual._pixel == reference._pixel
        assert actual._count == reference._count
        assert actual._phase == reference._phase
        assert actual.rng.bit_generator.state == reference.rng.bit_generator.state
