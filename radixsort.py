# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
#
#
# import numpy as np
#
# # %%
#
# data = np.random.randint(10, size=32)
#
# def radixargsort(data):
#     sorteddata = data.copy()
#     out = np.arange(len(data))
#     for i in range(data.dtype.itemsize * 8):
#         idx = np.bitwise_and(np.right_shift(sorteddata, i), 1).astype(bool)
#         out = np.concatenate((out[~idx], out[idx]))
#         sorteddata = np.concatenate((sorteddata[~idx], sorteddata[idx]))
#     return out
#
# %timeit radixargsort(np.random.randint(10, size=2048))
# %timeit np.argsort(np.random.randint(10, size=2048))
#
# # %%
#
# data = np.random.randint(10, size=32)
#
# sorteddata = data.copy()
# out = np.arange(len(data))
# buf = np.zeros((2, len(data)), dtype=data.dtype)
# argbuf = np.zeros((2, len(data)), dtype=data.dtype)
#
# for i in range(data.dtype.itemsize * 8):
#     x = [0, 0]
#     for j in range(len(sorteddata)):
#         n = (sorteddata[j] >> i) & 1
#         buf[n, x[n]] = sorteddata[j]
#         argbuf[n, x[n]] = out[j]
#         x[n] += 1
#     sorteddata = np.concatenate((buf[0, :x[0]], buf[1, :x[1]]))
#     out = np.concatenate((argbuf[0, :x[0]], argbuf[1, :x[1]]))
#
# np.allclose(sorteddata, np.sort(data))
#
# # %%

import cffi
import numpy as np

ffibuilder = cffi.FFI()


ffibuilder.cdef("""
    int cffi_radixargsort(long *, long *, long);
""")
ffibuilder.set_source("_radixargsort", r"""
    static long cffi_radixargsort(long *data, long *out, long n) {
        long i, d, r = 0;
        long t = sizeof(long);
        long x[2] = {0, 0};

        long *sorteddata=malloc(sizeof(*data) * n);
        for (i = 0; i < n; i++) {
            sorteddata[i] = data[i];
            out[i] = i;
        }

        long (*buf) = malloc(sizeof(*buf) * 2 * n);
        long (*argbuf) = malloc(sizeof(*argbuf) * 2 * n);

        for (d = 0; d < t * 8; d++) {
            x[0] = 0;
            x[1] = 0;

            for (i = 0; i < n; i++) {
                r = (data[i] >> d) & 1;
                buf[n * r + x[r]] = data[i];
                argbuf[n * r + x[r]] = out[i];
                x[r]++;
            }

            memcpy(data, buf, sizeof(*buf) * x[0]);
            memcpy(data + x[0], buf + n, sizeof(*buf) * x[1]);

            memcpy(out, argbuf, sizeof(*argbuf) * x[0]);
            memcpy(out + x[0], argbuf + n, sizeof(*argbuf) * x[1]);
        }
        free(sorteddata);
        free(buf);
        free(argbuf);
        return 0;
    }
""")
ffibuilder.compile(verbose=True)

import _radixargsort
data = np.random.randint(10, size=10)
out = np.zeros_like(data)

print(data)
print(out)

_radixargsort.lib.cffi_radixargsort(
    _radixargsort.ffi.cast("long *", data.ctypes.data),
    _radixargsort.ffi.cast("long *", out.ctypes.data),
    len(data),
)

print(data)
print(out)
