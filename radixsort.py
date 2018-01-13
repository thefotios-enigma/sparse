# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


import numpy as np
import cffi

def numpy_rargsort(data):
    sorteddata = data.copy()
    out = np.arange(len(data))
    for i in range(data.dtype.itemsize * 8):
        idx = np.bitwise_and(np.right_shift(sorteddata, i), 1).astype(bool)
        out = np.concatenate((out[~idx], out[idx]))
        sorteddata = np.concatenate((sorteddata[~idx], sorteddata[idx]))

    return out


def loop_rargsort(data):
    sorteddata = data.copy()
    out = np.arange(len(data))
    buf = np.zeros((2, len(data)), dtype=data.dtype)
    argbuf = np.zeros((2, len(data)), dtype=data.dtype)

    for i in range(data.dtype.itemsize * 8):
        x = [0, 0]
        for j in range(len(sorteddata)):
            n = (sorteddata[j] >> i) & 1
            buf[n, x[n]] = sorteddata[j]
            argbuf[n, x[n]] = out[j]
            x[n] += 1
        sorteddata = np.concatenate((buf[0, :x[0]], buf[1, :x[1]]))
        out = np.concatenate((argbuf[0, :x[0]], argbuf[1, :x[1]]))

    return out

ffibuilder = cffi.FFI()

ffibuilder.cdef("""
    long rargsort_long(long *, long *, long);
    long rargsort_longlong(long long *, long long *, long);
    long rargsort_int(int *, int *, long);
    long rargsort_short(short *, short *, long);
""")
ffibuilder.set_source("_radixargsort", r"""
    template <typename T> static long rargsort(T *data, T *out, long n) {
        long i, d, r = 0;
        long t = sizeof(T);
        long x[2] = {0, 0};

        T (*buf) = new T[2 * n];
        T (*argbuf) = new T[2 * n];

        for (d = 0; d < t * 8; d++) {
            x[0] = 0;
            x[1] = 0;

            for (i = 0; i < n; i++) {
                r = (data[i] >> d) & 1;
                buf[n * r + x[r]] = data[i];
                argbuf[n * r + x[r]] = out[i];
                x[r]++;
            }

            if(x[0] != 0 && x[1] != 0) {
                memcpy(data, buf, sizeof(*buf) * x[0]);
                memcpy(data + x[0], buf + n, sizeof(*buf) * x[1]);

                memcpy(out, argbuf, sizeof(*argbuf) * x[0]);
                memcpy(out + x[0], argbuf + n, sizeof(*argbuf) * x[1]);
            }
        }
        free(buf);
        free(argbuf);
        return 0;
    }

    static long rargsort_longlong(long long *data, long long *out, long n) {
        return rargsort(data, out, n);
    }

    static long rargsort_long(long *data, long *out, long n) {
        return rargsort(data, out, n);
    }

    static long rargsort_int(int *data, int *out, long n) {
        return rargsort(data, out, n);
    }

    static long rargsort_short(short *data, short *out, long n) {
        return rargsort(data, out, n);
    }
""", source_extension='.cpp', extra_compile_args=['-std=c++11'])
ffibuilder.compile(verbose=True)

def cffi_rargsort(data):
    import _radixargsort
    data = data.copy()
    out = np.arange(len(data))

    _radixargsort.lib.rargsort_long(
        _radixargsort.ffi.cast("long *", data.ctypes.data),
        _radixargsort.ffi.cast("long *", out.ctypes.data),
        len(data),
    )

    return out

data = np.random.randint(10, size=2048)

%timeit numpy_rargsort(data)
# 3.47 ms ± 119 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit loop_rargsort(data)
# 272 ms ± 8.96 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit cffi_rargsort(data)
# 814 µs ± 17.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
%timeit np.argsort(data)
# 39 µs ± 932 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
