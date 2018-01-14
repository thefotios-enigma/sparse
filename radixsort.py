# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


import numpy as np
import cffi
import timeit
import functools

def numpy_rargsort(data, base=1):
    sorteddata = data.copy()
    out = np.arange(len(data))
    for i in range(0, data.dtype.itemsize * 8, base):
        # Sort data into n buckets, depending on the n last bits
        idx = np.bitwise_and(np.right_shift(sorteddata, i), (2 ** base - 1))
        # Concatenate buckets together to get intermediate result
        sorteddata = np.concatenate([sorteddata[idx == b] for b in range(2 ** base)])
        out = np.concatenate([out[idx == b] for b in range(2 ** base)])

    return out

def loop_rargsort(data, base=1):
    sorteddata = data.copy()
    out = np.arange(len(data))
    buf = np.zeros((2 ** base, len(data)), dtype=data.dtype)
    argbuf = np.zeros((2 ** base, len(data)), dtype=data.dtype)

    for i in range(0, data.dtype.itemsize * 8, base):
        x = [0] * (2 ** base)
        # Sort data into n buckets, depending on the n last bits
        for j in range(len(sorteddata)):
            n = (sorteddata[j] >> i) & (2 ** base - 1)
            buf[n, x[n]] = sorteddata[j]
            argbuf[n, x[n]] = out[j]
            x[n] += 1
        # Concatenate buckets together to get intermediate result
        sorteddata = np.concatenate([buf[b, :x[b]] for b in range(2 ** base)])
        out = np.concatenate([argbuf[b, :x[b]] for b in range(2 ** base)])

    return out

ffibuilder = cffi.FFI()

ffibuilder.cdef("""
    long rargsort_long(long *, long *, long, int);
    long rargsort_longlong(long long *, long long *, long, int);
    long rargsort_int(int *, int *, long, int);
    long rargsort_short(short *, short *, long, int);
""")
ffibuilder.set_source("_radixargsort", r"""
    template <typename T> static long rargsort(T *data, T *out, long n, int base) {
        long i, d, r, tocopy, copied = 0;
        long pb = pow(2, base);
        long t = sizeof(T);
        long (*x) = new long[pb];

        T (*buf) = new T[pb * n];
        T (*argbuf) = new T[pb * n];

        for (d = 0; d < t * 8; d = d + base) {
            for (i = 0; i < pb; i++) {
                x[i] = 0;
            }

            // Sort data into n buckets, depending on the n last bits
            for (i = 0; i < n; i++) {
                r = (data[i] >> d) & (pb - 1);
                buf[n * r + x[r]] = data[i];
                argbuf[n * r + x[r]] = out[i];
                x[r]++;
            }

            // Concatenate buckets together to get intermediate result
            tocopy = 0;
            for (i = 0; i < pb; i++) {
                if (x[i] != 0) {
                    tocopy++;
                }
            }
            if(tocopy > 0) {
                copied = 0;
                for (i = 0; i < pb; i++) {
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

    static long rargsort_longlong(long long *data, long long *out, long n, int base) {
        return rargsort(data, out, n, base);
    }

    static long rargsort_long(long *data, long *out, long n, int base) {
        return rargsort(data, out, n, base);
    }

    static long rargsort_int(int *data, int *out, long n, int base) {
        return rargsort(data, out, n, base);
    }

    static long rargsort_short(short *data, short *out, long n, int base) {
        return rargsort(data, out, n, base);
    }
""", source_extension='.cpp', extra_compile_args=['-std=c++11'])
ffibuilder.compile(verbose=True)

def cffi_rargsort(data, base=1):
    import _radixargsort
    data = data.copy()
    out = np.arange(len(data))

    _radixargsort.lib.rargsort_long(
        _radixargsort.ffi.cast("long *", data.ctypes.data),
        _radixargsort.ffi.cast("long *", out.ctypes.data),
        len(data),
        base,
    )

    return out

# %%

data = np.random.randint(100, size=32784)

for base in [1, 2, 4, 8, 16]:
    print("---- ", base)
    out1 = loop_rargsort(data, base=base)
    out2 = numpy_rargsort(data, base=base)
    out3 = cffi_rargsort(data, base=base)
    assert np.allclose(out1, out2)
    assert np.allclose(out1, out3)
    print("loop   ", timeit.timeit(functools.partial(loop_rargsort, data, base=base), number=1))
    print("vector ", timeit.timeit(functools.partial(numpy_rargsort, data, base=base), number=1))
    print("cffi   ",timeit.timeit(functools.partial(cffi_rargsort, data, base=base), number=1))
print("----")
print("numpy  ", timeit.timeit(functools.partial(np.argsort, data), number=1))

# ----  1
# loop    12.137930491997395
# vector  0.06198880402371287
# cffi    0.02176301000872627
# ----  2
# loop    5.071827656007372
# vector  0.036922603962011635
# cffi    0.00828946998808533
# ----  4
# loop    1.6274524359614588
# vector  0.03659105201950297
# cffi    0.005959481990430504
# ----  8
# loop    0.9681051519582979
# vector  0.26732684002490714
# cffi    0.024591645982582122
# ----  16
# loop    1.2174103800207376
# vector  22.71097572200233
# cffi    0.009183672023937106
# ----
# numpy   0.002072302042506635
