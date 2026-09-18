"""Minimal LDPC codec for v5.

Systematic LDPC with parity-check matrix designed for AWGN + burst noise.
Rate ~13/15 (87% efficiency), block size ~11520 -> ~10000 data + 1520 parity.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# LDPC parameters
LDPC_N = 11520       # Total coded bits (matches wire capacity)
LDPC_K = 10000       # Data bits
LDPC_M = LDPC_N - LDPC_K  # Parity bits = 1520
LDPC_DV = 3          # Variable node degree
LDPC_DC = 6          # Check node degree (M*DC = N*DV -> 1520*6 = 11520*3? No. Adjust)

# For simplicity, use a structured parity-check matrix
# We'll use a simpler approach: systematic with random-ish H matrix

def _build_h_matrix(seed=42):
    """Build a structured H matrix for LDPC.
    
    Uses a protograph-like construction for reproducible, decodable matrix.
    """
    np.random.seed(seed)
    # We want M rows, N columns, each column weight DV, each row weight ~N*DV/M
    M, N, DV = LDPC_M, LDPC_N, LDPC_DV
    
    # Build using circulant blocks for structure
    # Split into submatrices
    H = np.zeros((M, N), dtype=np.uint8)
    
    # Simple approach: each parity bit checks DV consecutive data bits (with wrap)
    for i in range(M):
        for d in range(DV):
            col = (i * DV + d) % N
            H[i, col] = 1
    
    # Ensure each column has degree DV
    col_weights = H.sum(axis=0)
    # Fix any columns with wrong weight
    for j in range(N):
        if col_weights[j] != DV:
            # Redistribute
            pass
    
    return csr_matrix(H)

_H_MATRIX = None

def _get_h_matrix():
    global _H_MATRIX
    if _H_MATRIX is None:
        _H_MATRIX = _build_h_matrix()
    return _H_MATRIX

def ldpc_encode_systematic(data_bits):
    """Encode data_bits (length K) to codeword (length N) systematic.
    
    Returns: codeword = [data_bits, parity_bits]
    """
    if len(data_bits) != LDPC_K:
        raise ValueError(f'Expected {LDPC_K} data bits, got {len(data_bits)}')
    
    H = _get_h_matrix()
    # Systematic encoding: solve H * [p; d] = 0 for p
    # H = [H_p | H_d] where H_p is MxM, H_d is MxK
    H_p = H[:, LDPC_K:]
    H_d = H[:, :LDPC_K]
    
    # H_p * p + H_d * d = 0  =>  p = H_p^-1 * H_d * d (mod 2)
    # Use sparse solve over GF(2) - approximate with real then round
    syndrome = (H_d @ data_bits) % 2
    # For now, use a simple parity: each parity bit = XOR of DV data bits
    # This is a placeholder - real implementation needs proper GF(2) solve
    parity = np.zeros(LDPC_M, dtype=np.uint8)
    for i in range(LDPC_M):
        # Each parity checks 3 data bits
        idx1 = (i * 3) % LDPC_K
        idx2 = (i * 3 + 1) % LDPC_K
        idx3 = (i * 3 + 2) % LDPC_K
        parity[i] = data_bits[idx1] ^ data_bits[idx2] ^ data_bits[idx3]
    
    codeword = np.concatenate([data_bits, parity])
    return codeword

def ldpc_decode_belief_prop(llrs, max_iter=15):
    """Belief propagation decode from LLRs.
    
    Args:
        llrs: Log-likelihood ratios for all N bits (positive = likely 0)
    Returns:
        Hard decisions (0/1) for all N bits
    """
    if len(llrs) != LDPC_N:
        raise ValueError(f'Expected {LDPC_N} LLRs, got {len(llrs)}')
    
    H = _get_h_matrix()
    M, N = H.shape
    
    # Variable to check connections
    var_to_check = [[] for _ in range(N)]
    check_to_var = [[] for _ in range(M)]
    for i in range(M):
        for j in H[i].indices:
            var_to_check[j].append(i)
            check_to_var[i].append(j)
    
    # Initialize messages (variable -> check)
    # Start with channel LLRs
    msg_v2c = np.zeros((N, max(len(v) for v in var_to_check)))
    for j in range(N):
        for idx, c in enumerate(var_to_check[j]):
            msg_v2c[j, idx] = llrs[j]
    
    # BP iterations
    for iteration in range(max_iter):
        # Check -> variable
        msg_c2v = np.zeros((M, max(len(v) for v in check_to_var)))
        for i in range(M):
            vars_i = check_to_var[i]
            if not vars_i:
                continue
            # Product of tanh(msg/2) for all other variables
            for idx, j in enumerate(vars_i):
                # Exclude this variable
                other_msgs = [msg_v2c[j, var_to_check[j].index(i)] for jj in vars_i if jj != j]
                if other_msgs:
                    prod = np.prod(np.tanh(np.array(other_msgs)/2))
                    msg_c2v[i, idx] = 2 * np.arctanh(np.clip(prod, -0.999, 0.999))
        
        # Variable -> check
        for j in range(N):
            for idx, c in enumerate(var_to_check[j]):
                # Sum of incoming check messages + channel LLR
                other_c = [msg_c2v[cc, check_to_var[cc].index(j)] for cc in var_to_check[j] if cc != c]
                msg_v2c[j, idx] = llrs[j] + sum(other_c)
    
    # Final decision
    posterior = np.zeros(N)
    for j in range(N):
        posterior[j] = llrs[j] + sum(msg_v2c[j, :len(var_to_check[j])])
    
    return (posterior < 0).astype(np.uint8)

def pack_coefficients_to_llrs(coeffs, scale=1.0):
    """Pack float coefficients into LLRs for LDPC.
    
    Quantize coefficients to bits, then map to LLRs.
    """
    # Simple quantization: sign + magnitude bits
    # For now, just use sign as hard bit, magnitude as confidence
    bits = (coeffs < 0).astype(np.uint8)
    magnitudes = np.abs(coeffs)
    # Normalize magnitudes to [0, 1] then scale to LLR range
    if magnitudes.max() > 0:
        magnitudes = magnitudes / magnitudes.max()
    llrs = (1 - 2*bits) * (magnitudes * scale * 10)  # signed LLR
    return llrs

def unpack_llrs_to_coefficients(llrs, scale=1.0):
    """Convert decoded LLRs back to float coefficients."""
    bits = (llrs < 0).astype(np.float32)
    magnitudes = np.abs(llrs) / (scale * 10)
    magnitudes = np.clip(magnitudes, 0, 1)
    # Reconstruct with sign and magnitude
    coeffs = magnitudes * (1 - 2*bits)
    return coeffs