# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


import numpy as np
import cffi
import timeit
import functools

def numpy_rargsort(data, bits=1):
    base = 2 ** bits
    sorteddata = data.copy()
    out = np.arange(len(data))
    for i in range(0, data.dtype.itemsize * 8, bits):
        # Sort data into n buckets, depending on the n last bits
        idx = np.bitwise_and(np.right_shift(sorteddata, i), (base - 1))
        # Concatenate buckets together to get intermediate result
        sorteddata = np.concatenate([sorteddata[idx == b] for b in range(base)])
        out = np.concatenate([out[idx == b] for b in range(base)])

    return out

def loop_rargsort(data, bits=1):
    base = 2 ** bits
    sorteddata = data.copy()
    out = np.arange(len(data))
    buf = np.zeros((base, len(data)), dtype=data.dtype)
    argbuf = np.zeros((base, len(data)), dtype=data.dtype)

    for i in range(0, data.dtype.itemsize * 8, bits):
        x = [0] * (base)
        # Sort data into n buckets, depending on the n last bits
        for j in range(len(sorteddata)):
            n = (sorteddata[j] >> i) & (base - 1)
            buf[n, x[n]] = sorteddata[j]
            argbuf[n, x[n]] = out[j]
            x[n] += 1
        # Concatenate buckets together to get intermediate result
        sorteddata = np.concatenate([buf[b, :x[b]] for b in range(base)])
        out = np.concatenate([argbuf[b, :x[b]] for b in range(base)])

    return out

ffibuilder = cffi.FFI()

ffibuilder.cdef("""
    long rargsort_long(long *, long *, long, int);
    long rargsort_longlong(long long *, long long *, long, int);
    long rargsort_int(int *, int *, long, int);
    long rargsort_short(short *, short *, long, int);
""")
ffibuilder.set_source("_radixargsort", r"""
    template <typename T> static long rargsort(T *data, T *out, long n, int bits) {
        long i, d, r, tocopy, copied = 0;
        long base = pow(2, bits);
        long t = sizeof(T);
        long (*x) = new long[base];

        T (*buf) = new T[base * n];
        T (*argbuf) = new T[base * n];

        for (d = 0; d < t * 8; d = d + bits) {
            for (i = 0; i < base; i++) {
                x[i] = 0;
            }

            // Sort data into n buckets, depending on the n last bits
            for (i = 0; i < n; i++) {
                r = (data[i] >> d) & (base - 1);
                buf[n * r + x[r]] = data[i];
                argbuf[n * r + x[r]] = out[i];
                x[r]++;
            }

            // Concatenate buckets together to get intermediate result
            tocopy = 0;
            for (i = 0; i < base; i++) {
                if (x[i] != 0) {
                    tocopy++;
                }
            }
            if(tocopy > 0) {
                copied = 0;
                for (i = 0; i < base; i++) {
                    memcpy(data + copied, buf + n * i, sizeof(*buf) * x[i]);
                    memcpy(out + copied, argbuf + n * i, sizeof(*argbuf) * x[i]);
                    copied += x[i];
                }
            }
        }
        delete[] x;
        delete[] buf;
        delete[] argbuf;
        return 0;
    }

    static long rargsort_longlong(long long *data, long long *out, long n, int bits) {
        return rargsort(data, out, n, bits);
    }

    static long rargsort_long(long *data, long *out, long n, int bits) {
        return rargsort(data, out, n, bits);
    }

    static long rargsort_int(int *data, int *out, long n, int bits) {
        return rargsort(data, out, n, bits);
    }

    static long rargsort_short(short *data, short *out, long n, int bits) {
        return rargsort(data, out, n, bits);
    }
""", source_extension='.cpp', extra_compile_args=['-std=c++11'])
ffibuilder.compile(verbose=True)

import _radixargsort

def cffi_rargsort(data, bits=1):
    data = data.copy()
    out = np.arange(len(data))

    _radixargsort.lib.rargsort_long(
        _radixargsort.ffi.cast("long *", data.ctypes.data),
        _radixargsort.ffi.cast("long *", out.ctypes.data),
        len(data),
        bits,
    )

    return out

# %%

data = np.random.randint(100, size=8196)

for bits in [1, 2, 4, 8, 16]:
    print("---- ", bits)
    out1 = loop_rargsort(data, bits=bits)
    out2 = numpy_rargsort(data, bits=bits)
    out3 = cffi_rargsort(data, bits=bits)
    assert np.allclose(out1, out2)
    assert np.allclose(out1, out3)
    print("loop   ", timeit.timeit(functools.partial(loop_rargsort, data, bits=bits), number=10))
    print("vector ", timeit.timeit(functools.partial(numpy_rargsort, data, bits=bits), number=10))
    print("cffi   ",timeit.timeit(functools.partial(cffi_rargsort, data, bits=bits), number=10))
print("----")
print("numpy  ", timeit.timeit(functools.partial(np.argsort, data), number=10))

# ----  1
# loop    14.350793536985293
# vector  0.1265742700197734
# cffi    0.03426094597671181
# ----  2
# loop    5.145554315997288
# vector  0.10234820999903604
# cffi    0.01762533001601696
# ----  4
# loop    2.78461489797337
# vector  0.11544808198232204
# cffi    0.011012212955392897
# ----  8
# loop    1.5366798979812302
# vector  0.648971504997462
# cffi    0.05971593997674063
# ----  16
# loop    6.941754707018845
# vector  76.90049082605401
# cffi    0.09592188597889617
# ----
# numpy   0.006686180015094578
