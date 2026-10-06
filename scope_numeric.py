"""CPU numerical kernels shared by scope preparation, output and preview.

NumPy is array storage here; numerical operations execute in cached Numba.
Circular transforms use OpenCV's native CPU DFT, including non-power-of-two
trace lengths. No scope dependency on the modem's numerical helpers.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def finite_array(values):
    for value in values.flat:
        if not math.isfinite(value):
            return False
    return True


@njit(cache=True, nogil=True, fastmath=False)
def mass_search(cumulative,mark):
    return np.searchsorted(cumulative,mark,side='right')


@njit(cache=True, nogil=True, fastmath=False)
def circle_points(count,radius):
    out=np.empty((count,2),np.float64)
    for i in range(count):
        angle=(2.0*np.pi/count)*i
        out[i,0]=radius*np.cos(angle);out[i,1]=radius*np.sin(angle)
    return out


@njit(cache=True, nogil=True, fastmath=False)
def area_grid(image, rows, cols, single_precision=False):
    """Preserve the integer-bin area convention of the existing scope grid."""
    h, w = image.shape
    rows = max(1, min(rows, h)); cols = max(1, min(cols, w))
    # Match the legacy two cumulative passes, including their summation order.
    integral = np.zeros((h + 1, w + 1), np.float64)
    for x in range(w):
        value = 0.0
        for y in range(h):
            value += image[y, x]
            if single_precision:value=float(np.float32(value))
            integral[y + 1, x + 1] = value
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            value=integral[y,x]+integral[y,x-1]
            integral[y,x]=float(np.float32(value)) if single_precision else value
    out = np.empty((rows, cols), np.float64)
    for y in range(rows):
        y0 = y * h // rows; y1 = (y + 1) * h // rows
        for x in range(cols):
            x0 = x * w // cols; x1 = (x + 1) * w // cols
            value=integral[y1,x1]-integral[y0,x1]
            if single_precision:value=float(np.float32(value))
            value-=integral[y1,x0]
            if single_precision:value=float(np.float32(value))
            value+=integral[y0,x0]
            if single_precision:value=float(np.float32(value))
            out[y,x]=value/((y1-y0)*(x1-x0))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def importance_grid(image, gamma, trim, edge_gain, stochastic):
    h, w = image.shape
    clean = np.empty((h, w), np.float64)
    out = np.empty_like(clean)
    edge = np.zeros_like(clean)
    peak = 0.0
    floor = max(0.2 if stochastic else 0.0, trim)
    for y in range(h):
        for x in range(w):
            value = float(image[y, x])
            if math.isnan(value) or value < 0.0: value = 0.0
            elif value > 1.0: value = 1.0
            clean[y, x] = value
            out[y, x] = value ** max(gamma, 0.01) if value > floor else 0.0
    if edge_gain > 0.0 and h > 1 and w > 1:
        for y in range(h):
            for x in range(w):
                gx = (clean[y, min(x+1,w-1)] - clean[y, max(x-1,0)]) / (1 if x==0 or x==w-1 else 2)
                gy = (clean[min(y+1,h-1), x] - clean[max(y-1,0), x]) / (1 if y==0 or y==h-1 else 2)
                edge[y, x] = math.hypot(gx, gy)
                peak = max(peak, edge[y, x])
        if peak > 1e-12:
            for y in range(h):
                for x in range(w):
                    extra = edge_gain * edge[y, x] / peak
                    if stochastic: out[y, x] = max(out[y, x], extra * (0.25+0.75*clean[y,x]))
                    else: out[y, x] += extra
    mass = 0.0
    for y in range(h):
        for x in range(w):
            if stochastic: out[y, x] = min(1.0, max(0.0, out[y, x]))
            mass = max(mass, out[y,x]) if stochastic else mass + out[y,x]
    return out, mass


@njit(cache=True, nogil=True, fastmath=False)
def cumulative_mass(values):
    out = np.empty(values.size, np.float64)
    total = 0.0
    for i, value in enumerate(values.flat):
        total += value
        out[i] = total
    return out, total


@njit(cache=True, nogil=True, fastmath=False)
def systematic_samples(weights, points):
    cumulative, total = cumulative_mass(weights)
    if total <= 1e-12: return None
    count = max(8, points)
    indices = np.empty(count, np.int64); dwell = np.empty(count, np.int64)
    position = 0; found = 0
    for i in range(count):
        mark = (i + 0.5) * total / count
        while position < len(cumulative)-1 and cumulative[position] < mark:
            position += 1
        if found and indices[found-1] == position: dwell[found-1] += 1
        else:
            indices[found] = position; dwell[found] = 1; found += 1
    return indices[:found].copy(), dwell[:found].copy()


@njit(cache=True, nogil=True, fastmath=False)
def candidate_importance(luminance, edge, correction, gamma, trim, edge_gain):
    out = np.empty(len(luminance), np.float64)
    for i in range(len(out)):
        value = min(1.0, max(0.0, luminance[i]))
        weight = value ** max(gamma, 0.01) if value > max(0.0, trim) else 0.0
        if edge_gain > 0.0: weight += edge_gain * min(1.0, max(0.0, edge[i]))
        out[i] = weight * max(0.0, correction[i])
    return out


@njit(cache=True, nogil=True, fastmath=False)
def preview_rows(samples):
    values = np.empty(len(samples), np.float32)
    for i in range(len(samples)): values[i] = np.round(samples[i, 1], 5)
    values.sort()
    count = 0
    for i in range(len(values)):
        if i == 0 or values[i] != values[i-1]: count += 1
    return max(2, count)


@njit(cache=True, nogil=True, fastmath=False)
def _select_inplace(values, count, index):
    """Bounded in-place quickselect; avoid Numba partition's full-array copies."""
    low, high = 0, count - 1
    while low < high:
        pivot = values[(low + high) // 2]
        left, right = low, high
        while left <= right:
            while values[left] < pivot:
                left += 1
            while values[right] > pivot:
                right -= 1
            if left <= right:
                values[left], values[right] = values[right], values[left]
                left += 1
                right -= 1
        if index <= right:
            high = right
        elif index >= left:
            low = left
        else:
            break
    return values[index]


@njit(cache=True, nogil=True, fastmath=False)
def positive_percentile(values, percentile, threshold=0.0):
    scratch = np.empty(values.size, values.dtype)
    count = 0
    for value in values.flat:
        if value > threshold: scratch[count] = value; count += 1
    if count == 0: return 0.0, count
    # Selection avoids sorting millions of mostly-unneeded preview values.
    position = (count-1) * percentile / 100.0
    lo = int(position); hi = min(lo+1, count-1)
    lower = float(_select_inplace(scratch,count,lo))
    upper = float(_select_inplace(scratch,count,hi)) if hi != lo else lower
    result = lower + (upper-lower) * (position-lo)
    return result, count


@njit(cache=True, nogil=True, fastmath=False)
def array_percentile(values, percentile):
    ordered=values.ravel().copy()
    if not len(ordered):return np.nan
    for value in ordered:
        if np.isnan(value):return np.nan
    position=(len(ordered)-1)*percentile/100.0
    lo=int(position);hi=min(lo+1,len(ordered)-1)
    lower=float(_select_inplace(ordered,len(ordered),lo))
    upper=float(_select_inplace(ordered,len(ordered),hi)) if hi!=lo else lower
    return lower+(upper-lower)*(position-lo)


@njit(cache=True, nogil=True, fastmath=False)
def lit_bounds(values,threshold):
    h,w=values.shape;x0=w;y0=h;x1=y1=0
    for y in range(h):
        for x in range(w):
            if values[y,x]>threshold:
                x0=min(x0,x);y0=min(y0,y);x1=max(x1,x+1);y1=max(y1,y+1)
    return x0/w,y0/h,x1/w,y1/h


@njit(cache=True, nogil=True, fastmath=False)
def scale_clip(values, scale, low, high):
    out = np.empty_like(values)
    for i in range(values.size):
        out.flat[i] = min(high, max(low, values.flat[i] * scale))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def _spectral_gain(spectrum, samplerate, cutoff, order, kind, taper, max_boost):
    n = spectrum.shape[1]
    for k in range(n):
        frequency = min(k, n-k) * samplerate / n
        real = 1.0; imag = 0.0
        if kind == 0: # Butterworth magnitude, evaluated without overflow
            ratio = frequency / cutoff
            if ratio > 1.0:
                inverse = ratio ** (-order)
                real = inverse / math.sqrt(1.0+inverse*inverse)
            else: real = 1.0 / math.sqrt(1.0+ratio**(2*order))
        elif kind == 1:
            if taper > 0.0:
                x = min(1.0, max(0.0, (cutoff+taper-frequency)/taper))
                real = 0.5-0.5*math.cos(math.pi*x)
            else: real = 1.0 if frequency <= cutoff else 0.0
        else:
            if k == 0: real = 0.0
            else:
                imag = -cutoff/frequency if k <= n//2 else cutoff/frequency
                magnitude = math.hypot(real, imag)
                if magnitude > max_boost:
                    real *= max_boost/magnitude; imag *= max_boost/magnitude
            # A real Nyquist bin cannot carry an imaginary correction.
            if n % 2 == 0 and k == n//2: imag = 0.0
        for c in range(spectrum.shape[0]):
            a = spectrum[c,k,0]; b = spectrum[c,k,1]
            spectrum[c,k,0] = a*real-b*imag
            spectrum[c,k,1] = a*imag+b*real


def circular_filter(frame, samplerate, cutoff, order=4, kind=0, taper=0.0,
                    max_boost=8.0, dtype=np.float32):
    import cv2
    # Native OpenCV DFT accepts arbitrary trace sizes; never pad a repeating
    # frame, which would change its harmonics and seam semantics.
    rows = np.ascontiguousarray(frame.T, dtype=np.float64)
    spectrum = cv2.dft(rows, flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    _spectral_gain(spectrum, float(samplerate), float(cutoff), int(order),
                   int(kind), float(taper), float(max_boost))
    result = cv2.idft(spectrum, flags=cv2.DFT_ROWS | cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return np.ascontiguousarray(result.T, dtype=dtype)


@njit(cache=True, nogil=True, fastmath=False)
def stretch_grid(grid, low, high):
    out = np.empty_like(grid)
    for i in range(grid.size): out.flat[i] = min(1.0,max(0.0,(grid.flat[i]-low)/(high-low)))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def threshold_fraction(grid, threshold):
    count=0
    for value in grid.flat:
        if value>threshold: count+=1
    return count/grid.size


@njit(cache=True, nogil=True, fastmath=False)
def precondition_grid(grid, amount):
    out=np.empty_like(grid);h,w=grid.shape
    for y in range(h):
        for x in range(w):
            a=grid[y,max(0,x-2)];b=grid[y,max(0,x-1)];c=grid[y,x]
            d=grid[y,min(w-1,x+1)];e=grid[y,min(w-1,x+2)]
            blur=(a+4.0*b+6.0*c+4.0*d+e)/16.0
            out[y,x]=min(max(c+amount*(c-blur),min(b,c,d)),max(b,c,d))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def raster_points(grid, xs, ys, trim, floor, gamma, fields, field, reverse,
                  subcell, start, has_start):
    rows,cols=grid.shape
    points=np.empty((rows*cols,2),np.float32)
    weights=np.empty(rows*cols,np.float32);travel=np.zeros(rows*cols,np.bool_)
    used=0;segments=0;previous=start[0];has_previous=has_start
    first=field%fields
    sequence=np.arange(first,rows,fields)
    for index in range(len(sequence)):
        r=sequence[len(sequence)-1-index] if reverse else sequence[index]
        a=0;b=cols
        if trim>0:
            while a<cols and grid[r,a]<=trim: a+=1
            b=cols
            while b>a and grid[r,b-1]<=trim: b-=1
            if a==b:continue
        left=xs[a];right=xs[b-1]
        if subcell and trim>0 and b-a>1:
            step=xs[1]-xs[0]
            if a>0 and grid[r,a]>grid[r,a-1]:
                left-=step*min(1.0,max(0.0,(grid[r,a]-trim)/(grid[r,a]-grid[r,a-1])))
            if b<cols and grid[r,b-1]>grid[r,b]:
                right+=step*min(1.0,max(0.0,(grid[r,b-1]-trim)/(grid[r,b-1]-grid[r,b])))
        flip=abs(right-previous)<abs(left-previous) if has_previous else ((r%2==0) if reverse else (r%2==1))
        if used:
            weights[segments]=np.float32(1e-3);travel[segments]=True;segments+=1
        for i in range(b-a):
            x=b-1-i if flip else a+i
            value=left if x==a else (right if x==b-1 else xs[x])
            points[used,0]=value;points[used,1]=ys[r];used+=1
            if i>0:
                weights[segments]=max(grid[r,x],floor);segments+=1
        previous=float(points[used-1,0]);has_previous=True
    for i in range(segments):weights[i]=np.float32(max(weights[i],np.float32(1e-9))**gamma)
    return points[:used].copy(),weights[:segments].copy(),travel[:segments].copy()


@njit(cache=True, nogil=True, fastmath=False)
def live_luma(source,invert):
    h,w=source.shape[:2];out=np.empty((h,w),np.float32);peak=-np.inf
    for y in range(h):
        for x in range(w):
            if source.ndim==3:
                value=np.float32(0.0)
                for c in range(source.shape[2]):value=np.float32(value+source[y,x,c])
                value=np.float32(value/source.shape[2])
            else:value=source[y,x]
            out[y,x]=value
            if np.isnan(value):peak=np.nan
            elif not np.isnan(peak):peak=max(peak,value)
    for y in range(h):
        for x in range(w):
            value=out[y,x]
            if peak>1.5:value=np.float32(value/np.float32(255.0))
            if invert:
                if value<0:value=np.float32(0.0)
                elif value>1:value=np.float32(1.0)
                value=np.float32(1.0)-value
            out[y,x]=value
    return out


@njit(cache=True, nogil=True, fastmath=False)
def row_budgets(grid,reverse,trim,floor,gamma,samples):
    rows,cols=grid.shape;mass=np.zeros(rows,np.float64);total=0.0
    for i in range(rows):
        r=rows-1-i if reverse else i;first=True
        for x in range(cols):
            value=grid[r,x]
            if trim>0 and value<=trim:continue
            if first:first=False;continue
            mass[i]+=max(value,floor)**gamma
        total+=mass[i]
    out=np.empty(rows,np.int64)
    for i in range(rows):out[i]=max(2,int(np.round(mass[i]/total*samples))) if total>0 else max(2,samples//max(rows,1))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def sweep_row(grid,xs,ys,r,trim,floor,gamma,start,has_start):
    cols=grid.shape[1];a=0;b=cols
    if trim>0:
        while a<cols and grid[r,a]<=trim:a+=1
        while b>a and grid[r,b-1]<=trim:b-=1
    if a==b:return np.empty((0,2),np.float64),np.empty(0,np.float64)
    flip=abs(xs[b-1]-start[0])<abs(xs[a]-start[0]) if has_start else r%2==1
    count=b-a+int(has_start);points=np.empty((count,2),np.float64);weights=np.empty(max(0,count-1),np.float64)
    offset=0
    if has_start:points[0]=start;weights[0]=1e-3;offset=1
    for i in range(b-a):
        x=b-1-i if flip else a+i
        points[i+offset,0]=xs[x];points[i+offset,1]=ys[r]
        if i>0:weights[i+offset-1]=max(grid[r,x],floor)**gamma
    return points,weights


@njit(cache=True, nogil=True, fastmath=False)
def scale_float32(values,scale):
    out=np.empty(values.shape,np.float32)
    for i in range(values.size):out.flat[i]=values.flat[i]*scale
    return out


@njit(cache=True, nogil=True, fastmath=False)
def floor_weights(values):
    out=values.copy()
    for i in range(len(out)):
        if out[i]<1e-12:out[i]=1e-12
    return out


@njit(cache=True, nogil=True, fastmath=False)
def captured_luma(bgr):
    out=np.empty(bgr.shape[:2],np.float64)
    for y in range(bgr.shape[0]):
        for x in range(bgr.shape[1]):
            out[y,x]=(bgr[y,x,2]*0.2126+bgr[y,x,1]*0.7152+bgr[y,x,0]*0.0722)/255.0
    return out


@njit(cache=True, nogil=True, fastmath=False)
def gray_units(gray):
    out=np.empty(gray.shape,np.float32)
    for i in range(gray.size):out.flat[i]=np.float32(gray.flat[i])/np.float32(255.0)
    return out


@njit(cache=True, nogil=True, fastmath=False)
def luma_bytes(image):
    out=np.empty(image.shape,np.uint8)
    for i in range(image.size):
        value=image.flat[i]
        if np.isnan(value):value=0.0
        elif value<0.0:value=0.0
        elif value>1.0:value=1.0
        out.flat[i]=np.float32(np.float32(value)*np.float32(255.0))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def image_bytes(image):
    out=np.empty(image.shape,np.uint8)
    for i in range(image.size):
        value=image.flat[i]
        if np.isnan(value):value=0.0
        elif value<0.0:value=0.0
        elif value>255.0:value=255.0
        out.flat[i]=value
    return out


@njit(cache=True, nogil=True, fastmath=False)
def matte_mean(rgb):
    out=np.empty(rgb.shape[:2],np.uint8)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            out[y,x]=(int(rgb[y,x,0])+int(rgb[y,x,1])+int(rgb[y,x,2]))/3.0
    return out


@njit(cache=True, nogil=True, fastmath=False)
def baked_stipple_candidates(luminance,alpha,count):
    """Systematic proposal samples in the unchanged quantized bake format."""
    h,w=luminance.shape
    clean=np.empty((h,w),np.float64);edge=np.zeros((h,w),np.float64)
    for y in range(h):
        for x in range(w):clean[y,x]=float(luminance[y,x])/255.0
    peak=0.0
    if h>1 and w>1:
        for y in range(h):
            for x in range(w):
                gx=(clean[y,min(w-1,x+1)]-clean[y,max(0,x-1)])/(1 if x==0 or x==w-1 else 2)
                gy=(clean[min(h-1,y+1),x]-clean[max(0,y-1),x])/(1 if y==0 or y==h-1 else 2)
                edge[y,x]=np.hypot(gx,gy);peak=max(peak,edge[y,x])
        if peak>1e-12:
            for y in range(h):
                for x in range(w):edge[y,x]/=peak
    proposal=np.empty((h,w),np.float64)
    for y in range(h):
        for x in range(w):
            proposal[y,x]=(float(alpha[y,x])/255.0)*(0.15+0.75*clean[y,x]+0.10*edge[y,x])
    total=np.sum(proposal)
    xy=np.zeros((count,2),np.uint16);lae=np.zeros((count,3),np.uint8)
    if total<=1e-12:return xy,lae,np.float32(0.0)
    cumulative,_=cumulative_mass(proposal)
    position=0
    for i in range(count):
        mark=(i+0.5)*total/count
        while position<len(cumulative)-1 and cumulative[position]<mark:position+=1
        y=position//w;x=position%w
        xy[i,0]=np.round(x*65535.0/max(w-1,1));xy[i,1]=np.round(y*65535.0/max(h-1,1))
        lae[i,0]=luminance[y,x];lae[i,1]=alpha[y,x];lae[i,2]=np.round(edge[y,x]*255.0)
    return xy,lae,np.float32(total/proposal.size)


@njit(cache=True, nogil=True, fastmath=False)
def masked_quantiles(image,mask,bands):
    values=np.empty(image.size,np.float64);count=0
    for i in range(image.size):
        if mask.flat[i]>0:values[count]=image.flat[i];count+=1
    out=np.empty(bands if count else 0,np.float64)
    for i in range(len(out)):out[i]=array_percentile(values[:count],(100.0/(bands+1))*(i+1))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def normalized_points(points,width,height):
    out=np.empty(points.shape,np.float64);scale=max(width,height)/2.0
    for i in range(len(points)):
        out[i,0]=(points[i,0]-width/2.0)/scale
        out[i,1]=(points[i,1]-height/2.0)/scale
    return out


@njit(cache=True, nogil=True, fastmath=False)
def quantized_vertices(points):
    out=np.empty(points.shape,np.int16)
    for i in range(points.size):out.flat[i]=min(32767.0,max(-32767.0,np.round(points.flat[i]*32767.0)))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def moving_test_image(t):
    cx=80+45*np.sin(t/40.0);cy=60+30*np.cos(t/55.0)
    out=np.empty((120,160),np.float64)
    for y in range(120):
        for x in range(160):
            value=0.10+0.35*int((x//20+y//20)%2==0)
            value+=0.55*int((x-cx)**2+(y-cy)**2<260)
            out[y,x]=min(1.0,max(0.0,value))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def composite_thumbnail(main, floating, raw, invert):
    reference=main if main.size else floating
    h,w=reference.shape[:2];out=np.empty((h,w),np.float64)
    for y in range(h):
        for x in range(w):
            lm=am=lf=af=0.0
            if main.size:
                lm=float(main[y,x,2 if raw and main.shape[2]>=3 else 0])/255.0
                am=float(main[y,x,1])/255.0
            if floating.size:
                lf=float(floating[y,x,2 if raw and floating.shape[2]>=3 else 0])/255.0
                af=float(floating[y,x,1])/255.0
            value=lf*af+lm*am*(1.0-af)
            out[y,x]=min(1.0,max(0.0,af+am*(1.0-af)-value)) if invert else value
    return out


@njit(cache=True, nogil=True, fastmath=False)
def rotate_cloud(xy, turns):
    out=np.empty_like(xy)
    for i in range(len(xy)):
        u,v=xy[i]
        if turns==1:out[i,0]=v;out[i,1]=1.0-u
        elif turns==2:out[i,0]=1.0-u;out[i,1]=1.0-v
        else:out[i,0]=1.0-v;out[i,1]=u
    return out


@njit(cache=True, nogil=True, fastmath=False)
def trace_weights(image, points, gamma, trim, level):
    h,w=image.shape;m=float(max(h,w));extent=max(abs(level),1e-9)
    clean,_=sanitize_probability(image)
    out=np.zeros(len(points),np.float32)
    for i in range(len(points)):
        x=((points[i,0]/extent+1.0)*0.5*m-(m-w)*0.5)
        y=((1.0-points[i,1]/extent)*0.5*m-(m-h)*0.5)
        if not np.isfinite(x) or not np.isfinite(y) or x<0 or x>w-1 or y<0 or y>h-1:continue
        x0=int(x);y0=int(y);x1=min(x0+1,w-1);y1=min(y0+1,h-1);fx=x-x0;fy=y-y0
        value=((1-fx)*(1-fy)*clean[y0,x0]+fx*(1-fy)*clean[y0,x1]+
               (1-fx)*fy*clean[y1,x0]+fx*fy*clean[y1,x1])
        if value>max(0.0,trim):out[i]=value**max(0.01,gamma)
    return out


@njit(cache=True, nogil=True, fastmath=False)
def mux_positions(traces, weights, credit, phase, weighted):
    sources,count,_=traces.shape;out=np.empty((count,2),np.float32)
    row=np.empty(sources,np.float64)
    for i in range(count):
        selected=(phase+i)%sources
        if weighted:
            if sources==1:selected=0
            else:
                total=0.0
                for j in range(sources):
                    value=weights[i,j]
                    if np.isnan(value) or value<0:value=0.0
                    elif np.isinf(value):value=1.0
                    row[j]=value;total+=value
                inverse=1.0/total if total>1e-12 else 0.0
                for j in range(sources):credit[j]+=row[j]*inverse if total>1e-12 else 1.0/sources
                peak=np.max(credit);selected=phase
                for j in range(sources):
                    index=(phase+j)%sources
                    if credit[index]>=peak-1e-12:selected=index;break
                credit[selected]-=1.0;phase=(selected+1)%sources
        out[i]=traces[selected,i]
    if not weighted:phase=(phase+count)%sources
    return out,phase


@njit(cache=True, nogil=True, fastmath=False)
def overscan_path(points,weights,travel,overscan,travel_fraction):
    visible=1.0/overscan;count=0;content=0.0
    for i in range(len(points)-1):
        if i<len(travel) and travel[i]:count+=1
    for i in range(len(weights)):
        if not travel[i]:content+=weights[i]
    per_leg=content*travel_fraction/max(1.0-travel_fraction,1e-6)/max(count*3,1)
    out=np.empty((len(points)+2*count,2),np.float64)
    result=np.empty(len(out)-1,np.float64);offset=0
    out[0]=points[0]*visible
    for i in range(len(points)-1):
        if i<len(travel) and travel[i]:
            for index in (i,i+1):
                x=points[index,0]*visible;y=points[index,1]*visible
                if visible-abs(x)<=visible-abs(y):x=1.0 if x>=0 else -1.0
                else:y=1.0 if y>=0 else -1.0
                offset+=1;out[offset,0]=x;out[offset,1]=y;result[offset-1]=per_leg
            offset+=1;out[offset]=points[i+1]*visible;result[offset-1]=per_leg
        else:
            offset+=1;out[offset]=points[i+1]*visible;result[offset-1]=weights[i]
    return out,result,count


@njit(cache=True, nogil=True, fastmath=False)
def border_extension(points,weights,sx,sy,fraction):
    corners=np.array([[-sx,-sy],[sx,-sy],[sx,sy],[-sx,sy]],np.float32)
    nearest=0;best=np.inf
    for i in range(4):
        distance=np.hypot(corners[i,0]-points[-1,0],corners[i,1]-points[-1,1])
        if distance<best:best=distance;nearest=i
    loop=np.empty((5,2),np.float32);lengths=np.empty(4,np.float32)
    for i in range(5):loop[i]=corners[(nearest+i)%4]
    for i in range(4):lengths[i]=np.hypot(loop[i+1,0]-loop[i,0],loop[i+1,1]-loop[i,1])
    content=float(np.sum(weights));total=np.sum(lengths)
    border=content*fraction/(1.0-fraction)
    out=np.empty((len(points)+5,2),points.dtype);out[:len(points)]=points;out[len(points):]=loop
    result=np.empty(len(weights)+5,np.float64);result[:len(weights)]=weights
    result[len(weights)]=content*0.002
    for i in range(4):result[len(weights)+1+i]=np.float32(np.float32(lengths[i]/total)*np.float32(border))
    return out,result


@njit(cache=True, nogil=True, fastmath=False)
def alpha_samples(alpha,xy):
    scale=1.0
    for value in xy.flat:
        if value>1.0:scale=65535.0;break
    h,w=alpha.shape;out=np.empty(len(xy),np.float64)
    for i in range(len(xy)):
        x=min(w-1,max(0,int(np.round(xy[i,0]/scale*max(w-1,0)))))
        y=min(h-1,max(0,int(np.round(xy[i,1]/scale*max(h-1,0)))))
        out[i]=float(alpha[y,x])/255.0
    return out


@njit(cache=True, nogil=True, fastmath=False)
def candidate_cloud(xy,lae,mass,occlusion,invert):
    n=len(xy);uv=np.empty((n,2),np.float64)
    signal=np.empty(n,np.float64);edge=np.empty(n,np.float64);correction=np.empty(n,np.float64)
    for i in range(n):
        uv[i,0]=float(xy[i,0])/65535.0;uv[i,1]=float(xy[i,1])/65535.0
        luminance=float(lae[i,0])/255.0;alpha=float(lae[i,1])/255.0;ridge=float(lae[i,2])/255.0
        visible=alpha*(1.0-occlusion[i])
        signal[i]=(1.0-luminance)*visible if invert else luminance*visible
        edge[i]=ridge*visible
        proposal=alpha*(0.15+0.75*luminance+0.10*ridge)
        correction[i]=max(mass,0.0)/proposal if proposal>1e-9 else 0.0
    return uv,signal,edge,correction


@njit(cache=True, nogil=True, fastmath=False)
def baked_vertices(vertices,flip_y):
    out=np.empty(vertices.shape,np.float32)
    for i in range(len(vertices)):
        out[i,0]=np.float32(vertices[i,0])/np.float32(32767.0)
        out[i,1]=np.float32(vertices[i,1])/np.float32(32767.0)
        if flip_y:out[i,1]=-out[i,1]
    return out


@njit(cache=True, nogil=True, fastmath=False)
def nearest_path_vertex(points,position,closed):
    best=0;distance=np.inf
    for i in range(len(points)):
        if not closed and i!=0 and i!=len(points)-1:continue
        value=np.hypot(points[i,0]-position[0],points[i,1]-position[1])
        if value<distance:distance=value;best=i
    return distance,(best if closed or best==0 else -1)


@njit(cache=True, nogil=True, fastmath=False)
def subdivision_counts(points,max_segment):
    counts=np.empty(max(0,len(points)-1),np.int64);total=1;changed=False
    for i in range(len(counts)):
        length=np.hypot(points[i+1,0]-points[i,0],points[i+1,1]-points[i,1])
        counts[i]=max(1,int(np.ceil(length/max_segment)));total+=counts[i]
        if counts[i]>1:changed=True
    return counts,total,changed


@njit(cache=True, nogil=True, fastmath=False)
def subdivided_points(points,counts,total):
    out=np.empty((total,2),np.float64);out[0]=points[0];offset=1
    for i in range(len(counts)):
        for j in range(1,counts[i]+1):
            for axis in range(2):out[offset,axis]=points[i,axis]+(j/counts[i])*(points[i+1,axis]-points[i,axis])
            offset+=1
    return out


@njit(cache=True, nogil=True, fastmath=False)
def linear_axis(low,high,count):
    return np.linspace(low,high,count)


@njit(cache=True, nogil=True, fastmath=False)
def linear_trace(start,end,count):
    out=np.empty((count,2),np.float32)
    for i in range(count):
        t=i/max(1,count-1)
        for axis in range(2):out[i,axis]=float(start[axis])+t*(float(end[axis])-float(start[axis]))
    return out


@njit(cache=True, nogil=True, fastmath=False)
def yt_row(grid,xs,r,trim,floor,gamma,y):
    row=grid[r];cols=len(row);a=0;b=cols
    while a<cols and row[a]<=max(0.0,trim):a+=1
    while b>a and row[b-1]<=max(0.0,trim):b-=1
    points=np.empty((b-a,2),np.float64);weights=np.empty(max(0,b-a-1),np.float64)
    left=xs[a] if a<b else 0.0;right=xs[b-1] if a<b else 0.0
    if trim>0 and b-a>1:
        step=xs[1]-xs[0]
        if a>0 and row[a]>row[a-1]:left-=step*min(1.0,max(0.0,(row[a]-trim)/(row[a]-row[a-1])))
        if b<cols and row[b-1]>row[b]:right+=step*min(1.0,max(0.0,(row[b-1]-trim)/(row[b-1]-row[b])))
    for i in range(b-a):
        x=b-1-i if r%2 else a+i
        points[i,0]=left if x==a else (right if x==b-1 else xs[x]);points[i,1]=y
        if i>0:weights[i-1]=max(row[x],floor)**gamma
    return points,weights


@njit(cache=True, nogil=True, fastmath=False)
def clip_x(frame,low,high):
    for i in range(len(frame)):
        if frame[i,0]<low:frame[i,0]=low
        elif frame[i,0]>high:frame[i,0]=high
    return frame


@njit(cache=True, nogil=True, fastmath=False)
def argmax_first(values):
    best=0
    for i in range(1,len(values)):
        if values[i]>values[best]:best=i
    return best


@njit(cache=True, nogil=True, fastmath=False)
def pixel_positions(indices,width):
    out=np.empty((len(indices),2),np.float64)
    for i in range(len(indices)):
        out[i,0]=indices[i]%width;out[i,1]=indices[i]//width
    return out


@njit(cache=True, nogil=True, fastmath=False)
def pixel_route(pixels,width,height,level):
    maximum=float(max(width,height));out=np.empty_like(pixels)
    for i in range(len(pixels)):
        out[i,0]=(2.0*(pixels[i,0]+(maximum-width)*0.5)/maximum-1.0)*level
        out[i,1]=(1.0-2.0*(pixels[i,1]+(maximum-height)*0.5)/maximum)*level
    return out


@njit(cache=True, nogil=True, fastmath=False)
def cloud_route(points,aspect,level):
    sx=1.0 if aspect<=1.0 else 1.0/aspect;sy=aspect if aspect<=1.0 else 1.0
    out=np.empty_like(points)
    for i in range(len(points)):
        out[i,0]=(2.0*points[i,0]-1.0)*sx*level
        out[i,1]=(1.0-2.0*points[i,1])*sy*level
    return out


@njit(cache=True, nogil=True, fastmath=False)
def dwell_route(points,order,dwell):
    count=0
    for i in range(len(order)):count+=dwell[order[i]]
    out=np.empty((count,2),points.dtype);offset=0
    for i in range(len(order)):
        index=order[i]
        for j in range(dwell[index]):out[offset]=points[index];offset+=1
    return out


@njit(cache=True, nogil=True, fastmath=False)
def stipple_geometry(route,count,radius):
    out=np.empty((count,2),np.float32)
    for i in range(count):
        if len(route)==1:
            angle=2.0*np.pi*i/count
            out[i,0]=route[0,0]+radius*np.cos(angle)
            out[i,1]=route[0,1]+radius*np.sin(angle)
        else:
            t=((len(route)-1)/(count-1))*i
            low=int(t);high=min(low+1,len(route)-1);fraction=t-low
            for axis in range(2):out[i,axis]=route[low,axis]+fraction*(route[high,axis]-route[low,axis])
    return out


@njit(cache=True, nogil=True, fastmath=False)
def trace_border(source,count,aspect,level):
    aspect=max(aspect,1e-9)
    sx=level*(1.0 if aspect<=1.0 else 1.0/aspect)
    sy=level*(aspect if aspect<=1.0 else 1.0)
    corners=np.array([[-sx,-sy],[sx,-sy],[sx,sy],[-sx,sy]],np.float32)
    nearest=0;best=np.inf
    for i in range(4):
        dx=np.float32(corners[i,0]-source[-count-1,0])
        dy=np.float32(corners[i,1]-source[-count-1,1])
        value=np.float32(np.float32(dx*dx)+np.float32(dy*dy))
        if value<best:best=value;nearest=i
    lengths=np.empty(4,np.float64);total=0.0
    for i in range(4):
        a=(nearest+i)%4;b=(a+1)%4
        lengths[i]=np.float32(np.hypot(corners[b,0]-corners[a,0],corners[b,1]-corners[a,1]))
        total+=lengths[i]
    side_counts=np.ones(4,np.int64);remaining=count-5;fractions=np.empty(4,np.float64)
    used=0
    for i in range(4):
        exact=remaining*lengths[i]/total
        added=int(np.floor(exact));side_counts[i]+=added;used+=added;fractions[i]=exact-added
    for unused in range(remaining-used):
        selected=0
        for i in range(1,4):
            if fractions[i]>fractions[selected]:selected=i
        side_counts[selected]+=1;fractions[selected]=-1.0
    out=source.copy();offset=len(source)-count
    for side in range(4):
        a=corners[(nearest+side)%4];b=corners[(nearest+side+1)%4]
        for i in range(side_counts[side]):
            t=np.float32(np.float32(i)/side_counts[side])
            for axis in range(2):
                out[offset,axis]=np.float32(a[axis]+np.float32(t*np.float32(b[axis]-a[axis])))
            offset+=1
    return out


@njit(cache=True, nogil=True, fastmath=False)
def density_polyline(points,out):
    h,w=out.shape;maximum=float(max(w,h));xpad=(maximum-w)*0.5;ypad=(maximum-h)*0.5
    for i in range(len(points)-1):
        ax=(points[i,0]+1.0)*0.5*maximum-xpad
        ay=(1.0-points[i,1])*0.5*maximum-ypad
        bx=(points[i+1,0]+1.0)*0.5*maximum-xpad
        by=(1.0-points[i+1,1])*0.5*maximum-ypad
        steps=max(1,int(np.ceil(max(abs(bx-ax),abs(by-ay)))))
        for j in range(steps+1):
            t=j/steps;x=min(w-1,max(0,int(np.rint(ax+(bx-ax)*t))))
            y=min(h-1,max(0,int(np.rint(ay+(by-ay)*t))))
            out[y,x]=1.0


@njit(cache=True, nogil=True, fastmath=False)
def dilate_density(image,radius):
    h,w=image.shape;out=np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            value=0.0
            for j in range(max(0,y-radius),min(h,y+radius+1)):
                for i in range(max(0,x-radius),min(w,x+radius+1)):value=max(value,image[j,i])
            out[y,x]=value
    return out


@njit(cache=True, nogil=True, fastmath=False)
def polygon_crossings(points,vertices,crossings):
    for edge in range(len(vertices)):
        a=vertices[edge];b=vertices[(edge+1)%len(vertices)]
        for i in range(len(points)):
            px,py=points[i]
            if (a[1]>py)!=(b[1]>py):
                crossing=a[0]+(py-a[1])*(b[0]-a[0])/(b[1]-a[1])
                if px<crossing:crossings[i]+=1


@njit(cache=True, nogil=True, fastmath=False)
def segment_midpoints(points):
    out=np.empty((len(points)-1,2),points.dtype)
    for i in range(len(out)):
        for axis in range(2):out[i,axis]=0.5*(points[i,axis]+points[i+1,axis])
    return out


@njit(cache=True, nogil=True, fastmath=False)
def inside_parity(crossings):
    out=np.empty(len(crossings),np.bool_)
    for i in range(len(out)):out[i]=crossings[i]%2==1
    return out


@njit(cache=True, nogil=True, fastmath=False)
def outside_runs(inside):
    runs=np.empty((len(inside),2),np.int64);count=0;i=0
    while i<len(inside):
        if inside[i]:i+=1;continue
        first=i
        while i<len(inside) and not inside[i]:i+=1
        runs[count,0]=first;runs[count,1]=i+1;count+=1
    return runs[:count].copy()


@njit(cache=True, nogil=True, fastmath=False)
def nonnegative_values(values):
    out=np.empty_like(values)
    for i in range(values.size):
        value=values.flat[i]
        if np.isnan(value) or value<0.0:value=0.0
        elif np.isinf(value):value=1.0
        out.flat[i]=value
    return out


@njit(cache=True, nogil=True, fastmath=False)
def normalize_mass(values):
    out=np.empty_like(values);total=0.0
    for i in range(values.size):
        value=values.flat[i]
        if not np.isfinite(value) or value<0.0:value=0.0
        out.flat[i]=value;total+=value
    if total<=1e-12:return None
    for i in range(values.size):out.flat[i]/=total
    return out


@njit(cache=True, nogil=True, fastmath=False)
def combine_fields(fields):
    count,h,w=fields.shape;out=np.zeros((h,w),np.float64);peak=0.0
    for y in range(h):
        for x in range(w):
            value=0.0
            for k in range(count):value+=fields[k,y,x]
            out[y,x]=value/count;peak=max(peak,out[y,x])
    if peak<=1e-12:return None
    for y in range(h):
        for x in range(w):out[y,x]/=peak
    return out


@njit(cache=True, nogil=True, fastmath=False)
def retime_weighted(points,weights,travel_floor):
    value=nonnegative_values(weights);peak=np.max(value)
    if peak<=1e-12:return points
    count=len(points);distance=np.empty(count-1,np.float32)
    positive=np.empty(count-1,np.float32);used=0
    for i in range(count-1):
        distance[i]=np.hypot(points[i+1,0]-points[i,0],points[i+1,1]-points[i,1])
        if distance[i]>1e-9:positive[used]=distance[i];used+=1
    typical=np.median(positive[:used]) if used else np.inf
    floor=min(1.0,max(1e-4,travel_floor));duration=np.empty(count-1,np.float64)
    cumulative=np.zeros(count,np.float64)
    for i in range(count-1):
        dwell=0.5*(value[i]+value[i+1])/peak
        duration[i]=floor if distance[i]>4.0*typical else floor+(1.0-floor)*dwell
        cumulative[i+1]=cumulative[i]+duration[i]
    out=np.empty_like(points);segment=0
    for i in range(count):
        target=(cumulative[-1]/(count-1))*i
        while segment<count-2 and cumulative[segment+1]<=target:segment+=1
        fraction=(target-cumulative[segment])/max(duration[segment],1e-12)
        for axis in range(2):out[i,axis]=points[segment,axis]+fraction*(points[segment+1,axis]-points[segment,axis])
    out[0]=points[0];out[-1]=points[-1]
    return out


@njit(cache=True, nogil=True, fastmath=False)
def grid_axes(width,height,rows,cols):
    sx=1.0 if width>=height else width/float(height)
    sy=1.0 if height>=width else height/float(width)
    return np.linspace(-sx,sx,cols),-np.linspace(-sy,sy,rows)


@njit(cache=True, nogil=True, fastmath=False)
def sanitize_probability(values):
    out = np.empty_like(values); peak = 0.0
    for i in range(values.size):
        value = values.flat[i]
        if math.isnan(value) or value < 0.0: value = 0.0
        elif value > 1.0: value = 1.0
        out.flat[i] = value; peak = max(peak,value)
    return out, peak


@njit(cache=True, nogil=True, fastmath=False)
def stochastic_stream(prob, visited, rng, x, y, count, phase, samples, step,
                      reseed, radius, stride, level, start, handoff, cumulative, total):
    """Keep Generator draw order and visited/phase state identical to Python."""
    h,w = prob.shape; maximum = float(max(w,h))
    xpad=(maximum-w)*0.5; ypad=(maximum-h)*0.5
    out=np.empty((samples,2),np.float32); used_fallback=False; searches=0
    if samples == 0:
        return out,x,y,count,phase,used_fallback,searches
    begin=0
    if handoff:
        out[0]=start; begin=1
    for i in range(begin,samples):
        if phase>=1.0:
            if count%reseed==0:
                x=int(rng.integers(0,w)); y=int(rng.integers(0,h))
            if count%10==0: visited[:]=False
            sx,sy=x,y; direction=int(rng.integers(0,4)); found=False
            for arm in range(1,2*radius+1):
                for turn in range(2):
                    dx = 1 if direction==0 else (-1 if direction==2 else 0)
                    dy = 1 if direction==1 else (-1 if direction==3 else 0)
                    for move in range(arm):
                        sx+=stride*dx; sy+=stride*dy
                        if sx<0 or sx>=w or sy<0 or sy>=h: break
                        value=prob[sy,sx]
                        if value>0.0 and not visited[sy,sx] and rng.random()<value:
                            found=True; break
                    if found: break
                    direction=(direction+1)%4
                if found: break
            if found:
                x,y=sx,sy; visited[y,x]=True
            else:
                searches+=1
                for attempt in range(100):
                    sx=int(rng.integers(0,w)); sy=int(rng.integers(0,h)); value=prob[sy,sx]
                    if value>0.0 and rng.random()<value:
                        x,y=sx,sy; found=True; break
                if not found:
                    used_fallback=True
                    if len(cumulative)==0: cumulative,total=cumulative_mass(prob)
                    if total>1e-12:
                        flat=min(int(np.searchsorted(cumulative,rng.random()*total,side='right')),prob.size-1)
                        x=flat%w; y=flat//w
            count+=1; phase-=1.0
        out[i,0]=level*(2.0*(x+xpad)/maximum-1.0)
        out[i,1]=level*(1.0-2.0*(y+ypad)/maximum)
        phase+=step
    moving=False
    for i in range(1,samples):
        if out[i,0]!=out[i-1,0] or out[i,1]!=out[i-1,1]: moving=True; break
    if not moving:
        cx=float(out[-1,0]);cy=float(out[-1,1]);r=level/max(h,w)
        for i in range(samples):
            theta=2.0*np.pi*i/samples
            # The legacy idle-phase update is modulo samples, always zero.
            out[i,0]=np.float32(cx)+np.float32(np.float32(r)*np.float32(np.cos(theta)))
            out[i,1]=np.float32(cy)+np.float32(np.float32(r)*np.float32(np.sin(theta)))
    return out,x,y,count,phase,used_fallback,searches


_warmed = False


def warm_scope_numeric():
    global _warmed
    if _warmed: return
    for dtype in (np.float32, np.float64):
        image = np.zeros((2, 2), dtype=dtype)
        for readonly in (False, True):
            image.setflags(write=not readonly)
            area_grid(image, 2, 2)
            area_grid(image,2,2,dtype==np.float32)
            importance_grid(image, 2.0, 0.02, 0.0, False)
            finite_array(image)
            cumulative_mass(image)
            systematic_samples(image, 8)
            positive_percentile(image, 75.0)
            positive_percentile(image,75.0,0.01)
            array_percentile(image,50.0)
            lit_bounds(image,0.06)
            sanitize_probability(image)
            stretch_grid(image, 0.0, 1.0)
            threshold_fraction(image, 0.02)
            precondition_grid(image, 0.1)
            vector=np.zeros(2,dtype=dtype)
            vector.setflags(write=not readonly)
            finite_array(vector)
            cumulative_mass(vector)
            systematic_samples(vector,8)
        scale_clip(image, 1.0, -1.0, 1.0)
    area_grid(np.zeros((2,2), np.uint8), 2, 2)
    area_grid(np.zeros((2,2),np.uint8),2,2,False)
    values = np.zeros(2, np.float64)
    candidate_importance(values, values, values, 2.0, 0.02, 0.0)
    values.setflags(write=False)
    candidate_importance(values,values,values,2.0,0.02,0.0)
    preview_rows(np.zeros((2,2), np.float32))
    bgr=np.zeros((4,4,3),np.uint8)
    captured_luma(bgr)
    captured_luma(bgr[::2,::2])
    gray_units(np.zeros((2,2),np.uint8))
    gray=np.zeros((2,2),np.uint8)
    gray.setflags(write=False)
    gray_units(gray)
    luma_bytes(np.zeros((2,2),np.float32))
    moving_test_image(0)
    circle_points(4,0.7)
    mass_search(np.zeros(2,np.float64),0.0)
    rotate_cloud(np.zeros((2,2),np.float64),1)
    composite_thumbnail(np.zeros((2,2,2),np.uint8),np.empty((0,0,2),np.uint8),False,False)
    trace_weights(np.zeros((2,2),np.float64),np.zeros((2,2),np.float64),2.0,0.02,0.9)
    cached_image=np.zeros((2,2),np.float64);cached_image.setflags(write=False)
    trace_weights(cached_image,np.zeros((2,2),np.float64),2.0,0.02,0.9)
    mux_positions(np.zeros((2,2,2),np.float32),np.ones((2,2),np.float64),np.zeros(2,np.float64),0,True)
    nonnegative_values(np.zeros(2,np.float64))
    normalize_mass(np.zeros((2,2),np.float64))
    combine_fields(np.zeros((2,2,2),np.float64))
    retime_weighted(np.zeros((2,2),np.float32),np.ones(2,np.float64),0.03)
    grid_axes(2,2,2,2)
    points=np.zeros((2,2),np.float32)
    polygon_crossings(points,points,np.zeros(2,np.int64))
    segment_midpoints(points)
    inside_parity(np.zeros(2,np.int64))
    outside_runs(np.zeros(2,np.bool_))
    trace_border(np.zeros((16,2),np.float32),6,1.0,0.9)
    indices=np.zeros(2,np.int64);points=np.zeros((2,2),np.float64)
    argmax_first(indices)
    pixel_positions(indices,2)
    pixel_route(points,2,2,0.9)
    cloud_route(points,1.0,0.9)
    dwell_route(points,indices,np.ones(2,np.int64))
    stipple_geometry(points,2,0.01)
    density_polyline(points,np.zeros((2,2),np.float64))
    dilate_density(np.zeros((2,2),np.float64),1)
    linear_axis(-1.0,1.0,2)
    linear_trace(np.zeros(2,np.float32),np.zeros(2,np.float32),2)
    yt_row(np.zeros((2,2),np.float64),np.zeros(2,np.float64),0,0.02,0.012,2.2,0.0)
    clip_x(np.zeros((2,2),np.float64),-0.9,0.9)
    clip_x(np.zeros((4,2),np.float64)[::2],-0.9,0.9)
    live_luma(np.zeros((2,2),np.float32),False)
    live_luma(np.zeros((2,2,3),np.float32),False)
    row_budgets(np.zeros((2,2),np.float64),False,0.02,0.012,2.2,4)
    for dtype in (np.float32,np.float64):
        sweep_row(np.zeros((2,2),np.float64),np.zeros(2,dtype),np.zeros(2,dtype),
                  0,0.02,0.012,2.2,np.zeros(2,np.float64),False)
        scale_float32(np.zeros((2,2),dtype),0.9)
        scale_float32(np.zeros((4,2),dtype)[::2],0.9)
    floor_weights(np.zeros(2,np.float64))
    vertices=np.zeros((2,2),np.int16)
    baked_vertices(vertices,False)
    vertices.setflags(write=False)
    baked_vertices(vertices,False)
    for dtype in (np.float32,np.float64):
        points=np.zeros((2,2),dtype)
        nearest_path_vertex(points,np.zeros(2,np.float64),False)
        subdivision_counts(points,0.1)
        subdivided_points(points,np.ones(1,np.int64),2)
    alpha_samples(np.zeros((2,2),np.uint8),np.zeros((2,2),np.float64))
    candidate_cloud(np.zeros((2,2),np.uint16),np.zeros((2,3),np.uint8),1.0,np.zeros(2,np.float64),False)
    xy=np.zeros((2,2),np.uint16);lae=np.zeros((2,3),np.uint8)
    xy.setflags(write=False);lae.setflags(write=False)
    candidate_cloud(xy,lae,1.0,np.zeros(2,np.float64),False)
    for main_readonly in (False,True):
        for float_readonly in (False,True):
            main=np.zeros((2,2,2),np.uint8);floating=main.copy()
            main.setflags(write=not main_readonly);floating.setflags(write=not float_readonly)
            composite_thumbnail(main,floating,False,False)
    for order_readonly in (False,True):
        for dwell_readonly in (False,True):
            order=np.zeros(2,np.int64);dwell=np.ones(2,np.int64)
            order.setflags(write=not order_readonly);dwell.setflags(write=not dwell_readonly)
            argmax_first(dwell)
            pixel_positions(order,2)
            dwell_route(np.zeros((2,2),np.float64),order,dwell)
    overscan_path(np.zeros((2,2),np.float32),np.ones(1,np.float32),np.zeros(1,np.bool_),1.2,0.12)
    for dtype in (np.float32,np.float64):
        border_extension(np.zeros((2,2),dtype),np.ones(1,dtype),1.0,1.0,0.1)
    raster_points(np.zeros((2,2),np.float64),np.zeros(2,np.float64),np.zeros(2,np.float64),
                  0.02,0.012,2.2,1,0,False,True,np.zeros(2,np.float32),False)
    grid=np.zeros((2,2),np.float64);axis=np.zeros(2,np.float64)
    grid.setflags(write=False);axis.setflags(write=False)
    raster_points(grid,axis,axis,0.02,0.012,2.2,1,0,False,True,np.zeros(2,np.float32),False)
    _spectral_gain(np.zeros((2,4,2), np.float64), 48000.0, 2000.0, 4, 0, 0.0, 8.0)
    stochastic_stream(np.zeros((2,2),np.float64), np.zeros((2,2),np.bool_),
                      np.random.default_rng(0), 0, 0, 0, 1.0, 0, 1.0,
                      1, 1, 1, 0.9, np.zeros(2,np.float32), False,
                      np.empty(0,np.float64), 0.0)
    cached_cdf=np.zeros(4,np.float64);cached_cdf.setflags(write=False)
    stochastic_stream(np.zeros((2,2),np.float64),np.zeros((2,2),np.bool_),
                      np.random.default_rng(0),0,0,0,1.0,0,1.0,1,1,1,0.9,
                      np.zeros(2,np.float32),False,cached_cdf,0.0)
    _warmed = True
