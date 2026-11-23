# Serpentine fold: implemented equations

Implementation: `tools/v7_sk_fold.py::serpentine_encode_frame`,
`serpentine_decode_frame_exact`, `expand_exact`, and
`tools/v7_serpentine_codec.py` for coefficient placement and normalization.

## One folded slot

Let h be the normalized host, g the normalized guest, d the known dither
offset, Delta the host stair spacing, s the table's wire power scale, and C
the odd, piecewise-linear compander stored in the table. Its positive input
domain is [0,16] and output is [0,1]. The adaptive codec uses
Delta = 0.25 * Delta_table and leaves s unchanged.

```
q = round((h-d)/Delta)              # nearest-even rounding
p = +1 if q is even else -1
c = clip(C(g), -1+1e-12, 1-1e-12)
w = s * (d + Delta * (q + p*c/2))
```

For plain slots (table level 0), w = s*h; there is no guest. Folded slots
have table level 1. This table level is distinct from the host ladder index.

Decode a received symbol w_received:

```
v = w_received/s
q_hat = round((v-d)/Delta)
p_hat = +1 if q_hat is even else -1
r = clip(2*(v-d-q_hat*Delta)/Delta, -1, 1)
h_hat = d + q_hat*Delta
g_hat = C_inverse(p_hat*r)
```

The host is the cell midpoint, not the original continuous host. With a clean
symbol and no boundary slip, the guest is recovered by the inverse compander;
the host retains quantization error up to half a stair.

## Exact inverse

For c >= 0, find adjacent table entries T[j] <= c <= T[j+1], then

```
t = (c-T[j])/(T[j+1]-T[j])
C_inverse(c) = 16*(j+t)/(N-1)
```

Apply odd symmetry for negative c. At the compander endpoint the inverse
saturates at 16. This inverts the forward table's linear segments directly;
it is not a second uniformly sampled inverse lookup table.

## Adaptive normalization

The host ladder index is fixed at -1. With F = RATIO**(-1), a mono host
coefficient H is normalized as h = (H-mean)/(F*host_sd). A guest coefficient
G_j is normalized using its table standard deviation and activity scale A_j:

```
P = max_j,legs |G_j|/(guest_sd_j * F * A_j)
a = smallest member of {1,2,4,8,16} with P/a <= 16*(1-1e-6)
g_j = G_j/(guest_sd_j * F * A_j * a)
```

One a is selected across actual guest legs; the pilot signals it. If no factor
fits, encoding fails instead of silently clipping. Guest reconstruction
multiplies by a again. Host normalization and spacing are not enlarged by a.
Activity scaling uses the same midpoint stairs at sender and receiver.

## Stereo host/detail placement

For base H, detail D and table ride coefficient rho:

```
h_left  = (H-mean + rho*D)/(F*host_sd)
h_right = (H-mean - rho*D)/(F*host_sd)
H_hat = mean + F*host_sd*(h_left_hat+h_right_hat)/2
D_hat = F*host_sd*(h_left_hat-h_right_hat)/(2*rho)  # rho > 0
```

This is algebraic M/S reconstruction, without covariance shrink for the full
pair. Single-leg handling retains the existing stereo interpretation.

## Difference from legacy nested fold

The legacy one-guest encoder uses

```
w_legacy = s * (d + Delta*q + alpha*Delta*C(g))
```

It does not reverse guest orientation on alternating stairs. Its soft decoder
combines neighbouring cells and guest coordinates using noise likelihoods,
host-cell prior masses and guest density. A slip-free hard host estimate can
use the Laplace conditional cell mean rather than the midpoint. Legacy
decoding also uses an expanded inverse lookup table.

Serpentine instead uses alternating orientation, a half-cell guest span,
midpoint hosts and deterministic exact-table inversion. The paired host/detail
reconstruction is algebraic. It does not remove host quantization or guarantee
noise robustness: filtering/noise can change the residual or selected cell.

## Pushed adaptive math versus this integration

The pushed adaptive modes already used this serpentine/exact-inverse math.
This integration does not invent another mapping. It makes codec/table setup,
source preparation, experimental short-FIR emission and sample settings
accessible through the public path. Legacy profiles retain legacy math.
Kernel/source conversion and filtering are separate from the fold equations.
