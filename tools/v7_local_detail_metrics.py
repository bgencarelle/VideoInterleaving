"""Native Numba kernels for the localized-detail experiment.

NumPy names here are Numba array intrinsics: no interpreted NumPy/SciPy numeric
path, FFT library, BLAS solve, or Python fallback is used by these kernels.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def fft(values):
    n=len(values)
    if n==1:return values.copy()
    factor=2
    while n%factor and factor*factor<=n:factor+=1
    if n%factor:factor=n
    if factor==n:
        result=np.zeros(n,np.complex128)
        for k in range(n):
            for j in range(n):result[k]+=values[j]*np.exp(-2j*np.pi*k*j/n)
        return result
    part=n//factor;sub=np.empty((factor,part),np.complex128)
    for j in range(factor):sub[j]=fft(values[j::factor].copy())
    result=np.zeros(n,np.complex128)
    for k in range(n):
        for j in range(factor):result[k]+=sub[j,k%part]*np.exp(-2j*np.pi*j*k/n)
    return result


@njit(cache=True)
def band(image,frequency,direction='mixed'):
    rows,cols=image.shape;spectrum=np.empty((rows,cols),np.complex128)
    for y in range(rows):spectrum[y]=fft(image[y].astype(np.complex128))
    for x in range(cols):spectrum[:,x]=fft(spectrum[:,x].copy())
    for y in range(rows):
        fy=y if y<=rows//2 else y-rows
        for x in range(cols):
            fx=x if x<=cols//2 else x-cols
            radius=np.sqrt(fx*fx+fy*fy)
            outside=radius<frequency*.8 or radius>frequency*1.2
            if direction=='x' and abs(fy)>abs(fx)*.25:outside=True
            if direction=='y' and abs(fx)>abs(fy)*.25:outside=True
            if outside:spectrum[y,x]=0
    for x in range(cols):spectrum[:,x]=np.conj(fft(np.conj(spectrum[:,x]).copy()))/rows
    result=np.empty((rows,cols))
    for y in range(rows):result[y]=np.conj(fft(np.conj(spectrum[y]).copy())).real/cols
    return result


@njit(cache=True)
def recovery_values(truth,received):
    t=truth.ravel();r=received.ravel();energy=dot=received_energy=0.
    for i in range(len(t)):
        energy+=t[i]*t[i];dot+=t[i]*r[i];received_energy+=r[i]*r[i]
    if energy<1e-16:return False,0.,0.,0.,0.,False
    gain=dot/energy;correlation=dot/max(np.sqrt(energy*received_energy),1e-16)
    unexplained=error=0.
    for i in range(len(t)):
        unexplained+=(r[i]-gain*t[i])**2;error+=(r[i]-t[i])**2
    unexplained=np.sqrt(unexplained/energy);error=np.sqrt(error/energy)
    passed=correlation>=.8 and .5<=gain<=1.5 and unexplained<=.5 and error<=.5
    return True,correlation,gain,unexplained,error,passed


@njit(cache=True)
def gray_difference(a,b):
    rows,cols,_=a.shape;output=np.empty((rows,cols))
    for y in range(rows):
        for x in range(cols):output[y,x]=((a[y,x,0]-b[y,x,0])+(a[y,x,1]-b[y,x,1])+(a[y,x,2]-b[y,x,2]))/3
    return output


@njit(cache=True)
def audio_levels(audio):
    energy=peak=0.
    for value in audio.ravel():energy+=value*value;peak=max(peak,abs(value))
    return np.sqrt(energy/audio.size),peak


@njit(cache=True)
def attenuate(audio,rms_limit,peak_limit):
    """Turn down the whole waveform without changing its contents or timing."""
    rms,peak=audio_levels(audio)
    gain=1.
    if rms>0:gain=min(gain,rms_limit/rms)
    if peak>0:gain=min(gain,peak_limit/peak)
    # Leave one ppm of headroom for float32 sample rounding at the target.
    if gain<1:gain*=1-1e-6
    return audio*gain,gain


@njit(cache=True)
def image_mse(original,reconstruction):
    total=0.
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            for c in range(3):total+=(original[y,x,c]-reconstruction[y,x,c])**2
    return total/original.size


@njit(cache=True)
def detail_errors(expected,recovered,ranges,truth,actual):
    coordinates=np.zeros(expected.shape[1])
    for k in range(len(expected)):
        for j in range(len(coordinates)):
            coordinates[j]=max(coordinates[j],abs(expected[k,j]-recovered[k,j])/ranges[j])
    rms,_=audio_levels(truth);error,_=audio_levels(truth-actual)
    return coordinates,rms,error


@njit(cache=True)
def fixture_values(seed,family,frequency,load,contrast,width,height,direction='mixed'):
    np.random.seed(seed)
    detail=np.zeros((height,width))
    if family=='random-texture':
        for y in range(height):
            for x in range(width):detail[y,x]=np.random.normal()
    else:
        for k in range(load):
            cx=np.random.uniform(.1,.9);cy=np.random.uniform(.1,.9)
            theta=np.random.uniform(0,2*np.pi);phase=np.random.uniform(0,2*np.pi)
            if direction=='x':theta=0.
            elif direction=='y':theta=np.pi*.5
            sign=1 if np.random.random()>.5 else -1
            for y in range(height):
                for x in range(width):
                    xx=(x+.5)/width-cx;yy=(y+.5)/height-cy
                    u=xx*np.cos(theta)+yy*np.sin(theta);v=-xx*np.sin(theta)+yy*np.cos(theta)
                    if abs(v)>=.16 or abs(u)>=.22:continue
                    envelope=(.5+.5*np.cos(np.pi*v/.16))*(.5+.5*np.cos(np.pi*u/.22))
                    detail[y,x]+=envelope*(sign*np.tanh(u*frequency*8) if family=='edges' else np.cos(2*np.pi*frequency*u+phase))
    detail=band(detail,frequency,direction);maximum=0.
    for value in detail.ravel():maximum=max(maximum,abs(value))
    output=np.empty((height,width,3),np.uint8)
    for y in range(height):
        for x in range(width):
            base=.5+.06*np.cos(2*np.pi*2*(x+.5)/width)*np.sin(2*np.pi*3*(y+.5)/height)
            value=round((base+detail[y,x]*contrast/max(maximum,1e-12))*255)
            for c in range(3):output[y,x,c]=value
    return output


@njit(cache=True)
def image_bytes(image):
    output=np.empty(image.shape,np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(3):output[y,x,c]=int(min(1.,max(0.,image[y,x,c]))*255)
    return output


@njit(cache=True)
def unit_image(source):
    return source.astype(np.float64)/255
