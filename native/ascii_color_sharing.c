/* Small, reentrant kernels for optional terminal RGB colour sharing.
 * No Python ABI, compiler runtime, global mutable state or worker threads.
 * Build without fast-math/FMA contraction: DP tie-breaking is intentional.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

void ascii_lift(const uint8_t *rgb, const double *scale, uint8_t *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        unsigned v = rgb[3*i];
        if (rgb[3*i+1] > v) v = rgb[3*i+1];
        if (rgb[3*i+2] > v) v = rgb[3*i+2];
        for (unsigned c = 0; c < 3; c++) {
            int value = (int)(rgb[3*i+c] * scale[v] + .5);
            out[3*i+c] = (uint8_t)(value > 255 ? 255 : value);
        }
    }
}

/* Comparison features only: never reconstruct or modify source RGB here. */
void ascii_model420(const double *lab, double *model, size_t height,
                    size_t width, double edge)
{
    memcpy(model,lab,height*width*3*sizeof(double));
    for (size_t y=0; y<height; y+=2) for (size_t x=0; x<width; x+=2) {
        size_t cells[4], count=0;
        for (size_t dy=0; dy<2 && y+dy<height; dy++)
            for (size_t dx=0; dx<2 && x+dx<width; dx++) cells[count++]=(y+dy)*width+x+dx;
        int boundary=0;
        for (size_t i=0; i<count; i++) for (size_t j=i+1; j<count; j++) {
            double distance=0;
            for (unsigned c=0; c<3; c++) {
                double d=lab[cells[i]*3+c]-lab[cells[j]*3+c];
                distance+=d*d;
            }
            if (distance>edge*edge) boundary=1;
        }
        if (boundary) continue;
        double a=0,b=0;
        for (size_t i=0; i<count; i++) { a+=lab[cells[i]*3+1]; b+=lab[cells[i]*3+2]; }
        a/=count; b/=count;
        for (size_t i=0; i<count; i++) { model[cells[i]*3+1]=a; model[cells[i]*3+2]=b; }
    }
}

int ascii_share(const uint8_t *rgb, const double *lab, uint8_t *out,
                size_t height, size_t width, double tolerance, double edge,
                double lightness_limit, const double *edge_lab)
{
    size_t n = width + 1;
    double *cost = malloc(n * sizeof(double));
    double *loss = malloc(n * sizeof(double));
    size_t *parent = malloc(n * sizeof(size_t));
    size_t *representative = malloc(n * sizeof(size_t));
    if (!cost || !loss || !parent || !representative) {
        free(cost); free(loss); free(parent); free(representative);
        return -1;
    }
    memcpy(out, rgb, height * width * 3);
    const double radius2 = tolerance*tolerance + 1e-12;
    for (size_t y = 0; y < height; y++) {
        const double *row = lab + y*width*3;
        const double *edge_row = edge_lab + y*width*3;
        const uint8_t *colors = rgb + y*width*3;
        for (size_t i = 0; i < n; i++) {
            cost[i] = loss[i] = 1e30;
            parent[i] = representative[i] = 0;
        }
        cost[0] = loss[0] = 0;
        for (size_t start = 0; start < width; start++) {
            double total[3] = {0,0,0};
            double low[3], high[3];
            for (unsigned c = 0; c < 3; c++) low[c] = high[c] = row[3*start+c];
            for (size_t end = start; end < width; end++) {
                if (end > start) {
                    double jump = 0;
                    for (unsigned c = 0; c < 3; c++) {
                        double d = edge_row[3*end+c] - edge_row[3*(end-1)+c];
                        jump += d*d;
                    }
                    if (jump > edge*edge) break;
                }
                int impossible = 0;
                for (unsigned c = 0; c < 3; c++) {
                    double v = row[3*end+c];
                    total[c] += v;
                    if (v < low[c]) low[c] = v;
                    if (v > high[c]) high[c] = v;
                    double span = high[c]-low[c];
                    if (span*span > 4*radius2) impossible = 1;
                }
                /* Triangle inequality: a ball of radius r cannot contain two
                 * points more than 2r apart. Longer segments cannot fix this.
                 * Keep the original radius epsilon in this conservative bound.
                 */
                if (!impossible) {
                    for (size_t x = start; x < end; x++) {
                        double d2 = 0;
                        for (unsigned c = 0; c < 3; c++) {
                            double d = row[3*x+c]-row[3*end+c];
                            d2 += d*d;
                        }
                        if (d2 > 4*radius2) { impossible = 1; break; }
                    }
                }
                if (impossible) break;
                double min_L = -1e30, max_L = 1e30;
                if (lightness_limit >= 0) {
                    for (size_t x = start; x <= end; x++) {
                        double L = row[3*x];
                        double allowed = fmin(lightness_limit,
                            fmin(.02*fmax(0,L), .05*fmax(0,100-L)));
                        min_L = fmax(min_L, L-allowed);
                        max_L = fmin(max_L, L+allowed);
                    }
                    if (min_L > max_L+1e-12) break;
                }
                double count = (double)(end-start+1);
                double best = 1e30;
                size_t chosen = width;
                for (size_t x = start; x <= end; x++) {
                    if (row[3*x] < min_L-1e-12 || row[3*x] > max_L+1e-12) continue;
                    double d2 = 0;
                    for (unsigned c = 0; c < 3; c++) {
                        double d = row[3*x+c]-total[c]/count;
                        d2 += d*d;
                    }
                    if (d2 < best) { best = d2; chosen = x; }
                }
                if (chosen == width) continue;
                double worst = 0, squared = 0;
                int lightness_invalid = 0;
                for (size_t x = start; x <= end; x++) {
                    double d2 = 0;
                    for (unsigned c = 0; c < 3; c++) {
                        double d = row[3*x+c]-row[3*chosen+c];
                        d2 += d*d;
                    }
                    if (d2 > worst) worst = d2;
                    squared += d2;
                    if (lightness_limit >= 0) {
                        double L = row[3*x];
                        double allowed = fmin(lightness_limit,
                            fmin(.02*fmax(0,L), .05*fmax(0,100-L)));
                        if (fabs(L-row[3*chosen]) > allowed+1e-12)
                            lightness_invalid = 1;
                    }
                }
                if (worst > radius2 || lightness_invalid) continue;
                unsigned escape = 10;
                for (unsigned c = 0; c < 3; c++) {
                    unsigned v = colors[3*chosen+c];
                    escape += v >= 100 ? 3 : v >= 10 ? 2 : 1;
                }
                double candidate = cost[start] + escape;
                double candidate_loss = loss[start] + squared;
                if (candidate < cost[end+1] ||
                    (candidate == cost[end+1] && candidate_loss < loss[end+1])) {
                    cost[end+1] = candidate; loss[end+1] = candidate_loss;
                    parent[end+1] = start; representative[end+1] = chosen;
                }
            }
        }
        size_t end = width;
        while (end) {
            size_t start = parent[end], chosen = representative[end];
            for (size_t x = start; x < end; x++)
                memcpy(out+(y*width+x)*3, colors+chosen*3, 3);
            end = start;
        }
    }
    free(cost); free(loss); free(parent); free(representative);
    return 0;
}

static size_t number(uint8_t *out, size_t n, unsigned value)
{
    if (value >= 100) { out[n++] = '0'+value/100; out[n++] = '0'+(value/10)%10; }
    else if (value >= 10) out[n++] = '0'+value/10;
    out[n++] = '0'+value%10;
    return n;
}

size_t ascii_encode(const uint8_t *rgb, const uint32_t *chars, uint8_t *out,
                    size_t height, size_t width)
{
    size_t n = 0;
    for (size_t y = 0; y < height; y++) {
        if (y) { out[n++] = '\r'; out[n++] = '\n'; }
        uint32_t previous = UINT32_MAX;
        for (size_t x = 0; x < width; x++) {
            size_t i = y*width+x;
            unsigned r = rgb[3*i], g = rgb[3*i+1], b = rgb[3*i+2];
            uint32_t color = (r<<16)|(g<<8)|b;
            if (color != previous) {
                memcpy(out+n, "\033[38;2;", 7); n += 7;
                n = number(out,n,r); out[n++] = ';';
                n = number(out,n,g); out[n++] = ';';
                n = number(out,n,b); out[n++] = 'm';
                previous = color;
            }
            uint32_t cp = chars[i];
            /* NumPy U1 .tolist() exposes zero-filled cells as empty strings. */
            if (cp == 0) continue;
            if (cp < 128) out[n++] = cp;
            else if (cp < 2048) { out[n++] = 192|(cp>>6); out[n++] = 128|(cp&63); }
            else if (cp < 65536) {
                out[n++] = 224|(cp>>12); out[n++] = 128|((cp>>6)&63); out[n++] = 128|(cp&63);
            } else {
                out[n++] = 240|(cp>>18); out[n++] = 128|((cp>>12)&63);
                out[n++] = 128|((cp>>6)&63); out[n++] = 128|(cp&63);
            }
        }
    }
    return n;
}

static double distance2(const double *a, const double *b)
{
    double d0=a[0]-b[0], d1=a[1]-b[1], d2=a[2]-b[2];
    return d0*d0+d1*d1+d2*d2;
}

static unsigned digits(unsigned value)
{
    return value >= 100 ? 3 : value >= 10 ? 2 : 1;
}

/* Experimental byte-cost planner. Each row is a DAG of unchanged-cell skips
 * and single-colour draws. Candidates are CURRENT or PREVIOUS real RGB tints,
 * never a palette. Every visible output tint is bounded against CURRENT lab;
 * errors cannot accumulate relative to an already approximated prior frame.
 * The cost model charges CUP+SGR+UTF-8 bytes per draw. The final ANSI encoder
 * may merge draws further; the Python caller compares actual encoded costs.
 * Return 2 for invalid dimensions, 1 on allocation failure, otherwise 0.
 */
int ascii_temporal_plan(const uint8_t *rgb, const double *lab,
    const uint32_t *chars, const uint8_t *old, const double *old_lab,
    const uint32_t *old_chars, uint8_t *out, size_t height, size_t width,
    double tolerance, double edge)
{
    if (!width || width > 160 || !height || height > 60) return 2;
    size_t stride=width+1, maximum=2*width;
    size_t *stops=malloc(maximum*stride*sizeof(size_t));
    size_t *prefix=malloc(stride*sizeof(size_t));
    size_t *best_end=malloc(stride*sizeof(size_t));
    size_t *next=malloc(stride*sizeof(size_t));
    size_t *chosen=malloc(stride*sizeof(size_t));
    size_t *sources=malloc(maximum*sizeof(size_t));
    double *cost=malloc(stride*sizeof(double));
    if (!stops || !prefix || !best_end || !next || !chosen || !sources || !cost) {
        free(stops); free(prefix); free(best_end); free(next); free(chosen);
        free(sources); free(cost); return 1;
    }
    double radius2=tolerance*tolerance+1e-12, edge2=edge*edge;
    for (size_t y=0; y<height; y++) {
        size_t base=y*width, count=0;
        /* Source IDs [0,width) are current, [width,2*width) previous. */
        for (size_t s=0; s<maximum; s++) {
            const uint8_t *color=s<width ? rgb+3*(base+s) : old+3*(base+s-width);
            const uint32_t *glyph=s<width ? chars+base+s : old_chars+base+s-width;
            if (*glyph == 32) continue;
            int duplicate=0;
            for (size_t c=0; c<count; c++) {
                size_t t=sources[c];
                const uint8_t *other=t<width ? rgb+3*(base+t) : old+3*(base+t-width);
                if (!memcmp(color,other,3)) { duplicate=1; break; }
            }
            if (!duplicate) sources[count++]=s;
        }
        /* A spaces-only row still needs one drawing candidate to clear glyphs. */
        if (!count) sources[count++]=0;
        prefix[0]=0;
        for (size_t x=0; x<width; x++) {
            uint32_t cp=chars[base+x];
            prefix[x+1]=prefix[x]+(cp<128 ? 1 : cp<2048 ? 2 : cp<65536 ? 3 : 4);
        }
        for (size_t c=0; c<count; c++) {
            size_t s=sources[c];
            const double *feature=s<width ? lab+3*(base+s) : old_lab+3*(base+s-width);
            stops[c*stride+width]=width;
            for (size_t x=width; x-- > 0;) {
                if (chars[base+x]!=32 && distance2(feature,lab+3*(base+x))>radius2)
                    stops[c*stride+x]=x;
                else if (x+1<width &&
                    distance2(lab+3*(base+x),lab+3*(base+x+1))>edge2)
                    stops[c*stride+x]=x+1;
                else stops[c*stride+x]=stops[c*stride+x+1];
            }
        }
        cost[width]=0;
        for (size_t x=width; x-- > 0;) {
            cost[x]=1e30; next[x]=x+1; chosen[x]=maximum;
            if (chars[base+x]==old_chars[base+x] && (chars[base+x]==32 ||
                distance2(lab+3*(base+x),old_lab+3*(base+x))<=radius2))
                cost[x]=cost[x+1];
            size_t best=x+1;
            for (size_t e=x+1; e<=width; e++) {
                if (cost[e]+prefix[e] < cost[best]+prefix[best]) best=e;
                best_end[e]=best;
            }
            for (size_t c=0; c<count; c++) {
                size_t stop=stops[c*stride+x];
                if (stop<=x) continue;
                size_t e=best_end[stop], s=sources[c];
                const uint8_t *color=s<width ? rgb+3*(base+s) : old+3*(base+s-width);
                unsigned sgr=10+digits(color[0])+digits(color[1])+digits(color[2]);
                unsigned cup=4+digits((unsigned)y+1)+digits((unsigned)x+1);
                double candidate=cost[e]+prefix[e]-prefix[x]+sgr+cup;
                if (candidate<cost[x]) {
                    cost[x]=candidate; next[x]=e; chosen[x]=s;
                }
            }
        }
        for (size_t x=0; x<width;) {
            size_t e=next[x], s=chosen[x];
            if (s==maximum) {
                memcpy(out+3*(base+x),old+3*(base+x),3);
            } else {
                const uint8_t *color=s<width ? rgb+3*(base+s) : old+3*(base+s-width);
                for (size_t i=x; i<e; i++) memcpy(out+3*(base+i),color,3);
            }
            x=e;
        }
    }
    free(stops); free(prefix); free(best_end); free(next); free(chosen);
    free(sources); free(cost); return 0;
}
