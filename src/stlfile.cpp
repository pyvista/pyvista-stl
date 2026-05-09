/*
The MIT License (MIT)

Copyright (c) 2016 Aki Nyrhinen
Copyright (c) 2023-2024 Alex Kaszynski
Copyright (c) 2026 PyVista Developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define HAVE_MMAP 1
#else
#define HAVE_MMAP 0
#endif

#include "array_support.h"
#include "hash96.h"
#include "stlfile.h"

namespace nb = nanobind;

// Format tags returned by detect_format.
enum StlFormat : int { STL_INVALID = 0, STL_ASCII = 1, STL_BINARY = 2 };

// Binary STL is little-endian on disk. We memcpy 4-byte float words
// directly from the mapped file into uint32_t slots, which is only
// correct on little-endian hosts. Fail the build on big-endian rather
// than silently mis-parsing.
static_assert(
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
#elif defined(_WIN32) || defined(_M_IX86) || defined(_M_X64) ||                \
    defined(_M_ARM) || defined(_M_ARM64)
    true,
#else
    false,
#endif
    "pyvista-stl supports little-endian platforms only.");

// ---------------------------------------------------------------------------
// File mapping helpers
// ---------------------------------------------------------------------------

struct MappedFile {
  const uint8_t *data = nullptr;
  size_t size = 0;
#if HAVE_MMAP
  int fd = -1;
  void *mmap_addr = nullptr;
  size_t mmap_len = 0;
#else
  uint8_t *owned = nullptr;
#endif

  bool open(const char *path) {
#if HAVE_MMAP
    fd = ::open(path, O_RDONLY);
    if (fd < 0)
      return false;
    struct stat st;
    if (fstat(fd, &st) != 0) {
      ::close(fd);
      fd = -1;
      return false;
    }
    size = static_cast<size_t>(st.st_size);
    if (size == 0) {
      ::close(fd);
      fd = -1;
      return true; // size==0 means empty mapping
    }
    mmap_len = size;
    mmap_addr = ::mmap(nullptr, mmap_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_addr == MAP_FAILED) {
      mmap_addr = nullptr;
      ::close(fd);
      fd = -1;
      return false;
    }
    // Hint the kernel we'll read sequentially so it can prefetch.
    ::madvise(mmap_addr, mmap_len, MADV_SEQUENTIAL);
    ::madvise(mmap_addr, mmap_len, MADV_WILLNEED);
    data = static_cast<const uint8_t *>(mmap_addr);
    return true;
#else
    FILE *fp = fopen(path, "rb");
    if (!fp)
      return false;
    if (fseek(fp, 0, SEEK_END) != 0) {
      fclose(fp);
      return false;
    }
    long long_size = ftell(fp);
    if (long_size < 0) {
      fclose(fp);
      return false;
    }
    size = static_cast<size_t>(long_size);
    rewind(fp);
    if (size + 1 < size) { // overflow guard
      fclose(fp);
      return false;
    }
    owned = (uint8_t *)malloc(size + 1);
    if (!owned) {
      fclose(fp);
      return false;
    }
    if (fread(owned, 1, size, fp) != size) {
      free(owned);
      owned = nullptr;
      fclose(fp);
      return false;
    }
    owned[size] = 0;
    fclose(fp);
    data = owned;
    return true;
#endif
  }

  ~MappedFile() {
#if HAVE_MMAP
    if (mmap_addr)
      ::munmap(mmap_addr, mmap_len);
    if (fd >= 0)
      ::close(fd);
#else
    free(owned);
#endif
  }
};

// ---------------------------------------------------------------------------
// Resource caps
// ---------------------------------------------------------------------------

// Default upper bound on the declared triangle count. Real-world meshes
// well below this; the cap exists to bound peak memory under malicious
// or corrupted input. Override with PYVISTA_STL_MAX_TRIS.
static constexpr triangle_t DEFAULT_MAX_TRIS = 200u * 1000u * 1000u;

static triangle_t resolve_max_tris() {
  triangle_t cap = DEFAULT_MAX_TRIS;
  if (const char *env = std::getenv("PYVISTA_STL_MAX_TRIS")) {
    long long v = std::strtoll(env, nullptr, 10);
    if (v > 0) {
      uint64_t v64 = (uint64_t)v;
      if (v64 > (uint64_t)~(triangle_t)0)
        v64 = (uint64_t)~(triangle_t)0;
      cap = (triangle_t)v64;
    }
  }
  return cap;
}

// Shrink a malloc-allocated buffer in place. Returns the (possibly
// moved) pointer on success.
//
// ``realloc(p, 0)`` is implementation-defined: on glibc it frees ``p``
// and returns NULL, which would leave the caller with a dangling
// pointer if we treated NULL as "shrink failed, keep the old
// pointer". Handle the zero-size case explicitly, and on a non-zero
// shrink that genuinely fails, keep the original allocation valid.
template <typename T> static T *try_shrink(T *p, size_t new_count) {
  if (new_count == 0) {
    std::free(p);
    return nullptr;
  }
  void *q = std::realloc(p, new_count * sizeof(T));
  if (!q)
    return p;
  return (T *)q;
}

// Compute nextpow2 of (multiplier * ntris) without 32-bit overflow.
// Saturates at 1u << 31 (the largest nextpow2 representable in
// uint32_t). The caller must already have rejected ntris values that
// exceed the configured cap.
static uint32_t saturating_nextpow2(uint64_t multiplier, triangle_t ntris) {
  uint64_t target = multiplier * (uint64_t)ntris;
  if (target >= (1ull << 31))
    return 1u << 31;
  return nextpow2((uint32_t)target);
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

static StlFormat detect_format(const uint8_t *data, size_t size) {
  if (size < 15)
    return STL_INVALID;
  // ASCII header begins with "solid " (case-sensitive in spec).
  if (size >= 6 && std::memcmp(data, "solid ", 6) == 0) {
    // Some binary writers also start with "solid "; verify via size match
    // for binary: header(80) + count(4) + 50*ntris.
    if (size >= 84) {
      uint32_t nTriangles;
      std::memcpy(&nTriangles, data + 80, sizeof(nTriangles));
      if (size == 84 + (size_t)nTriangles * 50)
        return STL_BINARY;
    }
    return STL_ASCII;
  }
  if (size < 84)
    return STL_INVALID;
  uint32_t nTriangles;
  std::memcpy(&nTriangles, data + 80, sizeof(nTriangles));
  if (size != 84 + (size_t)nTriangles * 50)
    return STL_INVALID;
  return STL_BINARY;
}

// ---------------------------------------------------------------------------
// Sequential vertex hashtable insert (small files / fallback).
// ---------------------------------------------------------------------------

static inline vertex_t seq_vertex(uint32_t *verts, vertex_t &nverts,
                                  vertex_t *vht, vertex_t vhtcap,
                                  const uint32_t *vert) {
  vertex_t hash = final96(vert[0], vert[1], vert[2]);
  vertex_t mask = vhtcap - 1;
  for (vertex_t i = 0; i < vhtcap; i++) {
    vertex_t *vip = vht + ((hash + i) & mask);
    vertex_t vi = *vip;
    if (vi == 0) {
      // claim id = nverts
      verts[3 * nverts + 0] = vert[0];
      verts[3 * nverts + 1] = vert[1];
      verts[3 * nverts + 2] = vert[2];
      *vip = nverts + 1;
      return nverts++;
    }
    vi--;
    const uint32_t *p = verts + 3 * vi;
    if (p[0] == vert[0] && p[1] == vert[1] && p[2] == vert[2])
      return vi;
  }
  return ~(vertex_t)0;
}

// ---------------------------------------------------------------------------
// Concurrent vertex hashtable insert (parallel binary path).
// ---------------------------------------------------------------------------

static constexpr uint32_t SLOT_RESERVED = 0xFFFFFFFFu;

static inline vertex_t mt_vertex(uint32_t *verts, std::atomic<uint32_t> *vht,
                                 vertex_t vhtcap,
                                 std::atomic<uint32_t> &nverts_atomic,
                                 const uint32_t *vert) {
  vertex_t hash = final96(vert[0], vert[1], vert[2]);
  vertex_t mask = vhtcap - 1;
  for (vertex_t i = 0; i < vhtcap; i++) {
    std::atomic<uint32_t> &slot = vht[(hash + i) & mask];
    for (;;) {
      uint32_t cur = slot.load(std::memory_order_acquire);
      if (cur == 0) {
        uint32_t expected = 0;
        if (slot.compare_exchange_strong(expected, SLOT_RESERVED,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
          uint32_t id = nverts_atomic.fetch_add(1, std::memory_order_relaxed);
          verts[3 * id + 0] = vert[0];
          verts[3 * id + 1] = vert[1];
          verts[3 * id + 2] = vert[2];
          slot.store(id + 1, std::memory_order_release);
          return id;
        }
        // CAS failed; expected now holds new value
        continue;
      }
      if (cur == SLOT_RESERVED) {
        // Another thread claimed; spin briefly.
        // Pause to ease the bus.
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#endif
        continue;
      }
      // occupied
      uint32_t id = cur - 1;
      const uint32_t *p = verts + 3 * id;
      if (p[0] == vert[0] && p[1] == vert[1] && p[2] == vert[2])
        return id;
      break; // collision: try next probe
    }
  }
  return ~(vertex_t)0;
}

// ---------------------------------------------------------------------------
// Binary STL reader (single-threaded).
// ---------------------------------------------------------------------------

static int loadstl_binary_seq(const uint8_t *data, size_t /*size*/,
                              triangle_t ntris, float **vertp, vertex_t *nvertp,
                              vertex_t **trip, triangle_t *ntrip) {
  // Allocate generously; max unique == 3*ntris.
  uint32_t *verts =
      (uint32_t *)malloc((size_t)3 * ntris * 3 * sizeof(uint32_t));
  vertex_t *tris = (vertex_t *)malloc((size_t)ntris * 3 * sizeof(vertex_t));

  // Hashtable capacity must be > 3*ntris worst-case (every vertex unique).
  // Sizing to nextpow2(4*ntris) caps the worst-case load factor at 25%,
  // matching the original libstl behavior. Use the saturating helper to
  // avoid 32-bit overflow when ntris is near the configured cap.
  vertex_t vhtcap = saturating_nextpow2(4, ntris);
  if (vhtcap < 1024)
    vhtcap = 1024;
  vertex_t *vht = (vertex_t *)calloc(vhtcap, sizeof(vertex_t));
  if (!verts || !tris || !vht) {
    free(verts);
    free(tris);
    free(vht);
    return -1;
  }

  const uint8_t *base = data + 84;
  vertex_t nverts = 0;
  for (triangle_t i = 0; i < ntris; i++) {
    const uint8_t *trec = base + (size_t)i * 50;
    // Skip 12-byte normal (trec[0..11]).
    // Three 12-byte vertices follow at trec+12.
    for (int ti = 0; ti < 3; ti++) {
      uint32_t vert[3];
      std::memcpy(vert, trec + 12 + 12 * ti, 12);
      vertex_t vi = seq_vertex(verts, nverts, vht, vhtcap, vert);
      if (vi == ~(vertex_t)0) {
        free(vht);
        free(verts);
        free(tris);
        return -1;
      }
      tris[3 * i + ti] = vi;
    }
    // attribute byte count (trec+48..49) intentionally ignored.
  }
  free(vht);
  verts = try_shrink(verts, (size_t)nverts * 3);
  *vertp = (float *)verts;
  *nvertp = nverts;
  *trip = tris;
  *ntrip = ntris;
  return 0;
}

// ---------------------------------------------------------------------------
// Binary STL reader (multi-threaded with concurrent hashtable).
// ---------------------------------------------------------------------------

// Allocate a (zeroed) atomic-uint32 hashtable backed by anonymous mmap when
// available, with transparent-hugepage hint to reduce TLB pressure on the
// random hashtable accesses. Falls back to operator new[].
static std::atomic<uint32_t> *alloc_atomic_table(size_t n, void *&backing,
                                                 size_t &backing_len) {
  backing = nullptr;
  backing_len = 0;
#if HAVE_MMAP
  size_t bytes = n * sizeof(std::atomic<uint32_t>);
  void *p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p != MAP_FAILED) {
    ::madvise(p, bytes, MADV_HUGEPAGE);
    backing = p;
    backing_len = bytes;
    return reinterpret_cast<std::atomic<uint32_t> *>(p);
  }
#endif
  return new (std::nothrow) std::atomic<uint32_t>[n]();
}

static void free_atomic_table(std::atomic<uint32_t> *p, void *backing,
                              size_t backing_len) {
#if HAVE_MMAP
  if (backing) {
    ::munmap(backing, backing_len);
    return;
  }
#else
  (void)backing;
  (void)backing_len;
#endif
  delete[] p;
}

static int loadstl_binary_mt(const uint8_t *data, size_t /*size*/,
                             triangle_t ntris, unsigned nthreads, float **vertp,
                             vertex_t *nvertp, vertex_t **trip,
                             triangle_t *ntrip) {
  uint32_t *verts =
      (uint32_t *)malloc((size_t)3 * ntris * 3 * sizeof(uint32_t));
  vertex_t *tris = (vertex_t *)malloc((size_t)ntris * 3 * sizeof(vertex_t));

  // Hashtable capacity. Worst case is 3*ntris unique vertices; for
  // well-merged meshes the unique count is closer to 0.5*ntris.
  // Sizing to nextpow2(ntris) gives a 4x reduction over the worst-case
  // bound and a much smaller working set in cache and TLB. If a
  // degenerate input saturates the table, mt_vertex returns ~0 and the
  // caller falls back to the sequential path which uses the full bound.
  vertex_t vhtcap = saturating_nextpow2(1, ntris);
  if (vhtcap < 1024)
    vhtcap = 1024;
  void *vht_backing = nullptr;
  size_t vht_backing_len = 0;
  std::atomic<uint32_t> *vht =
      alloc_atomic_table(vhtcap, vht_backing, vht_backing_len);
  if (!verts || !tris || !vht) {
    free(verts);
    free(tris);
    free_atomic_table(vht, vht_backing, vht_backing_len);
    return -1;
  }

  std::atomic<uint32_t> nverts_atomic{0};
  std::atomic<int> error_flag{0};
  const uint8_t *base = data + 84;

  vertex_t mask = vhtcap - 1;
  auto worker = [&](triangle_t start, triangle_t end) {
    const uint8_t *trec_end = base + (size_t)end * 50;
    const uint8_t *trec = base + (size_t)start * 50;
    // Prefetch distance (in triangles ahead). Tuned empirically.
    constexpr triangle_t PF_DIST = 16;
    for (triangle_t i = start; i < end; i++, trec += 50) {
      // Prefetch upcoming raw triangle data and the hashtable slots
      // it will hash into, to overlap DRAM latency.
      const uint8_t *pf_trec = trec + (size_t)PF_DIST * 50;
      if (pf_trec < trec_end) {
        __builtin_prefetch(pf_trec, 0, 0);
        // Hash the prefetched vertices to prefetch their slots.
        for (int ti = 0; ti < 3; ++ti) {
          uint32_t pv[3];
          std::memcpy(pv, pf_trec + 12 + 12 * ti, 12);
          uint32_t h = final96(pv[0], pv[1], pv[2]);
          __builtin_prefetch(&vht[h & mask], 1, 0);
        }
      }
      for (int ti = 0; ti < 3; ti++) {
        uint32_t vert[3];
        std::memcpy(vert, trec + 12 + 12 * ti, 12);
        vertex_t vi = mt_vertex(verts, vht, vhtcap, nverts_atomic, vert);
        if (vi == ~(vertex_t)0) {
          error_flag.store(1, std::memory_order_relaxed);
          return;
        }
        tris[3 * i + ti] = vi;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  triangle_t per = (ntris + nthreads - 1) / nthreads;
  for (unsigned t = 0; t < nthreads; t++) {
    triangle_t s = t * per;
    triangle_t e = s + per;
    if (e > ntris)
      e = ntris;
    if (s >= e)
      break;
    threads.emplace_back(worker, s, e);
  }
  for (auto &th : threads)
    th.join();

  free_atomic_table(vht, vht_backing, vht_backing_len);

  if (error_flag.load()) {
    free(verts);
    free(tris);
    return -1;
  }

  vertex_t nverts = nverts_atomic.load();
  verts = try_shrink(verts, (size_t)nverts * 3);
  *vertp = (float *)verts;
  *nvertp = nverts;
  *trip = tris;
  *ntrip = ntris;
  return 0;
}

// ---------------------------------------------------------------------------
// ASCII STL reader (single-pass, fast atof, mmap input).
// ---------------------------------------------------------------------------

// Parse a signed decimal float with optional exponent. Faster than
// strtof for the well-behaved values found in STL files because it
// avoids locale lookups and the strtof rounding-mode handshake.
static inline float fast_atof(const char *&p, const char *end) {
  static const double pow10_pos[] = {
      1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,
      1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
      1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29,
      1e30, 1e31, 1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38,
  };
  static const double pow10_neg[] = {
      1e0,   1e-1,  1e-2,  1e-3,  1e-4,  1e-5,  1e-6,  1e-7,  1e-8,  1e-9,
      1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19,
      1e-20, 1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26, 1e-27, 1e-28, 1e-29,
      1e-30, 1e-31, 1e-32, 1e-33, 1e-34, 1e-35, 1e-36, 1e-37, 1e-38,
  };

  bool neg = false;
  if (p < end && *p == '-') {
    neg = true;
    ++p;
  } else if (p < end && *p == '+') {
    ++p;
  }

  // 18 decimal digits is the largest count guaranteed to fit in
  // uint64_t (max value 1.8e19). Past that we still consume the digits
  // (so the parser advances past the number) but stop accumulating
  // into mant; the trailing digits are folded into the exponent.
  static constexpr int MANT_DIGIT_CAP = 18;
  uint64_t mant = 0;
  int mant_digits = 0;
  int dec_digits = 0;
  int dropped_digits = 0;
  bool seen_dot = false;
  while (p < end) {
    unsigned c = (unsigned char)*p;
    if (c >= '0' && c <= '9') {
      if (mant_digits < MANT_DIGIT_CAP) {
        mant = mant * 10 + (c - '0');
        mant_digits++;
        if (seen_dot)
          dec_digits++;
      } else if (!seen_dot) {
        dropped_digits++;
      }
      ++p;
    } else if (c == '.' && !seen_dot) {
      seen_dot = true;
      ++p;
    } else {
      break;
    }
  }

  int exp_val = 0;
  if (p < end && (*p == 'e' || *p == 'E')) {
    ++p;
    bool eneg = false;
    if (p < end && *p == '-') {
      eneg = true;
      ++p;
    } else if (p < end && *p == '+') {
      ++p;
    }
    while (p < end && *p >= '0' && *p <= '9') {
      exp_val = exp_val * 10 + (*p - '0');
      ++p;
    }
    if (eneg)
      exp_val = -exp_val;
  }

  // Integer digits dropped past the cap shift the implicit exponent up.
  int eff = exp_val - dec_digits + dropped_digits;
  double result = (double)mant;
  if (eff > 0) {
    if (eff > 38)
      eff = 38;
    result *= pow10_pos[eff];
  } else if (eff < 0) {
    int e = -eff;
    if (e > 38)
      e = 38;
    result *= pow10_neg[e];
  }
  if (neg)
    result = -result;
  return (float)result;
}

static inline uint32_t f_to_u32(float f) {
  uint32_t r;
  std::memcpy(&r, &f, 4);
  return r;
}

// Treat both LF and CR as line terminators so the parser handles LF,
// CRLF, and rare CR-only files (some classic-Mac-era exporters).
static inline bool is_eol(uint8_t c) { return c == '\n' || c == '\r'; }
static inline bool is_eol(char c) { return is_eol(static_cast<uint8_t>(c)); }

static int loadstl_ascii(const uint8_t *data, size_t size, float **vertp,
                         vertex_t *nvertp, vertex_t **trip, triangle_t *ntrip) {
  const char *ptr = (const char *)data;
  const char *end = (const char *)data + size;

  // Skip "solid ..." line
  while (ptr < end && !is_eol(*ptr))
    ptr++;
  while (ptr < end && is_eol(*ptr))
    ptr++;

  // Estimate triangle count from file size: each facet is roughly 200 bytes.
  size_t ntris_estimate = (size / 180) + 1024;
  size_t tris_cap = ntris_estimate;
  size_t verts_cap = tris_cap * 3;
  vertex_t *tris = (vertex_t *)malloc(tris_cap * 3 * sizeof(vertex_t));
  uint32_t *verts = (uint32_t *)malloc(verts_cap * 3 * sizeof(uint32_t));
  vertex_t vhtcap = nextpow2((uint32_t)(verts_cap * 2));
  if (vhtcap < 1024)
    vhtcap = 1024;
  vertex_t *vht = (vertex_t *)calloc(vhtcap, sizeof(vertex_t));
  if (!tris || !verts || !vht) {
    free(tris);
    free(verts);
    free(vht);
    return -1;
  }

  auto skip_ws = [&]() {
    while (ptr < end) {
      unsigned c = (unsigned char)*ptr;
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        ++ptr;
      else
        break;
    }
  };
  auto skip_to_eol = [&]() {
    while (ptr < end && !is_eol(*ptr))
      ptr++;
    while (ptr < end && is_eol(*ptr))
      ptr++;
  };

  triangle_t ntris = 0;
  vertex_t nverts = 0;
  uint32_t v[3][3];
  int v_idx = 0;

  while (ptr < end) {
    skip_ws();
    if (ptr >= end)
      break;
    if (end - ptr >= 6 && std::memcmp(ptr, "vertex", 6) == 0) {
      ptr += 6;
      skip_ws();
      float x = fast_atof(ptr, end);
      skip_ws();
      float y = fast_atof(ptr, end);
      skip_ws();
      float z = fast_atof(ptr, end);
      // skip rest of line
      while (ptr < end && !is_eol(*ptr))
        ptr++;
      while (ptr < end && is_eol(*ptr))
        ptr++;
      if (v_idx < 3) {
        v[v_idx][0] = f_to_u32(x);
        v[v_idx][1] = f_to_u32(y);
        v[v_idx][2] = f_to_u32(z);
        v_idx++;
      }
      continue;
    }
    if (end - ptr >= 8 && std::memcmp(ptr, "endfacet", 8) == 0) {
      ptr += 8;
      skip_to_eol();
      if (v_idx == 3) {
        if ((size_t)ntris >= tris_cap) {
          size_t new_cap = tris_cap * 2;
          void *q = std::realloc(tris, new_cap * 3 * sizeof(vertex_t));
          if (!q) {
            free(vht);
            free(tris);
            free(verts);
            return -1;
          }
          tris = (vertex_t *)q;
          tris_cap = new_cap;
        }
        vertex_t vi[3];
        for (int k = 0; k < 3; k++) {
          if ((size_t)nverts + 1 >= verts_cap) {
            size_t new_cap = verts_cap * 2;
            void *q = std::realloc(verts, new_cap * 3 * sizeof(uint32_t));
            if (!q) {
              free(vht);
              free(tris);
              free(verts);
              return -1;
            }
            verts = (uint32_t *)q;
            verts_cap = new_cap;
          }
          vi[k] = seq_vertex(verts, nverts, vht, vhtcap, v[k]);
          if (vi[k] == ~(vertex_t)0) {
            // Hashtable saturated. Double its capacity and rehash.
            vertex_t old_cap = vhtcap;
            vertex_t *old_vht = vht;
            vhtcap *= 2;
            vht = (vertex_t *)calloc(vhtcap, sizeof(vertex_t));
            if (!vht) {
              free(old_vht);
              free(tris);
              free(verts);
              return -1;
            }
            vertex_t mask = vhtcap - 1;
            for (vertex_t idx = 0; idx < old_cap; idx++) {
              vertex_t v_old = old_vht[idx];
              if (v_old == 0)
                continue;
              const uint32_t *vo = verts + 3 * (v_old - 1);
              vertex_t h = final96(vo[0], vo[1], vo[2]);
              for (vertex_t j = 0; j < vhtcap; j++) {
                vertex_t *p = vht + ((h + j) & mask);
                if (*p == 0) {
                  *p = v_old;
                  break;
                }
              }
            }
            free(old_vht);
            vi[k] = seq_vertex(verts, nverts, vht, vhtcap, v[k]);
            if (vi[k] == ~(vertex_t)0) {
              free(vht);
              free(tris);
              free(verts);
              return -1;
            }
          }
        }
        tris[3 * ntris + 0] = vi[0];
        tris[3 * ntris + 1] = vi[1];
        tris[3 * ntris + 2] = vi[2];
        ntris++;
      }
      // Always reset, whether or not the facet was committed. Without
      // this, vertex state from a malformed facet (v_idx != 3) leaks
      // into the next facet's parse.
      v_idx = 0;
      continue;
    }
    skip_to_eol();
  }

  free(vht);
  verts = try_shrink(verts, (size_t)nverts * 3);
  tris = try_shrink(tris, (size_t)ntris * 3);
  *vertp = (float *)verts;
  *nvertp = nverts;
  *trip = tris;
  *ntrip = ntris;
  return 0;
}

// ---------------------------------------------------------------------------
// ASCII STL reader (multi-threaded, concurrent hashtable).
// ---------------------------------------------------------------------------

// Find the byte offset of the next "facet " line at or after p.
// "facet " is at the start of a possibly-indented line. memchr is
// SIMD-accelerated in glibc and bypasses what was a startup hot spot
// in the multi-GB ASCII profile.
//
// For LF and CRLF inputs (the common case) memchr('\n') finds the
// terminator. For rare CR-only inputs memchr('\n') returns NULL and
// we fall back to scanning for CR. Doing the LF scan first avoids the
// O(N^2) trap of scanning the whole buffer twice on every iteration.
static const uint8_t *find_next_facet(const uint8_t *p, const uint8_t *end) {
  while (p < end) {
    size_t remaining = (size_t)(end - p);
    const uint8_t *nl = (const uint8_t *)std::memchr(p, '\n', remaining);
    if (!nl)
      nl = (const uint8_t *)std::memchr(p, '\r', remaining);
    if (!nl)
      return end;
    // Consume any sequence of EOL chars (handles CR, LF, CRLF).
    const uint8_t *q = nl;
    while (q < end && is_eol(*q))
      ++q;
    while (q < end && (*q == ' ' || *q == '\t'))
      ++q;
    if (end - q >= 6 && std::memcmp(q, "facet ", 6) == 0)
      return q;
    p = nl + 1;
  }
  return end;
}

// Parse a contiguous ASCII region [p, pend) that begins exactly at a
// "facet " line. Vertices are deduplicated via the shared verts/vht
// concurrent hashtable; triangle indices are written to local_tris.
// Returns the number of triangles written, or ~0 on overflow.
static triangle_t parse_ascii_chunk(const uint8_t *p, const uint8_t *pend,
                                    uint32_t *verts, std::atomic<uint32_t> *vht,
                                    vertex_t vhtcap,
                                    std::atomic<uint32_t> &nverts_atomic,
                                    vertex_t *local_tris,
                                    triangle_t local_cap) {
  triangle_t ntris = 0;
  uint32_t v[3][3];
  int v_idx = 0;
  const uint8_t *end = pend;

  while (p < end) {
    // Skip leading whitespace
    while (p < end) {
      uint8_t c = *p;
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        ++p;
      else
        break;
    }
    if (p >= end)
      break;

    if (end - p >= 6 && std::memcmp(p, "vertex", 6) == 0) {
      p += 6;
      while (p < end && (*p == ' ' || *p == '\t'))
        ++p;
      const char *cp = (const char *)p;
      const char *cend = (const char *)end;
      float x = fast_atof(cp, cend);
      while (cp < cend && (*cp == ' ' || *cp == '\t'))
        ++cp;
      float y = fast_atof(cp, cend);
      while (cp < cend && (*cp == ' ' || *cp == '\t'))
        ++cp;
      float z = fast_atof(cp, cend);
      p = (const uint8_t *)cp;
      while (p < end && !is_eol(*p))
        ++p;
      while (p < end && is_eol(*p))
        ++p;
      if (v_idx < 3) {
        v[v_idx][0] = f_to_u32(x);
        v[v_idx][1] = f_to_u32(y);
        v[v_idx][2] = f_to_u32(z);
        ++v_idx;
      }
      continue;
    }
    if (end - p >= 8 && std::memcmp(p, "endfacet", 8) == 0) {
      p += 8;
      while (p < end && !is_eol(*p))
        ++p;
      while (p < end && is_eol(*p))
        ++p;
      if (v_idx == 3) {
        if (ntris >= local_cap)
          return ~(triangle_t)0;
        vertex_t vi[3];
        for (int k = 0; k < 3; k++) {
          vi[k] = mt_vertex(verts, vht, vhtcap, nverts_atomic, v[k]);
          if (vi[k] == ~(vertex_t)0)
            return ~(triangle_t)0;
        }
        local_tris[3 * ntris + 0] = vi[0];
        local_tris[3 * ntris + 1] = vi[1];
        local_tris[3 * ntris + 2] = vi[2];
        ++ntris;
      }
      // Always reset (see loadstl_ascii for rationale).
      v_idx = 0;
      continue;
    }
    // Skip unrecognized tokens to end of line
    while (p < end && !is_eol(*p))
      ++p;
    while (p < end && is_eol(*p))
      ++p;
  }
  return ntris;
}

static int loadstl_ascii_mt(const uint8_t *data, size_t size, unsigned nthreads,
                            float **vertp, vertex_t *nvertp, vertex_t **trip,
                            triangle_t *ntrip) {
  const uint8_t *base = data;
  const uint8_t *end = data + size;

  // Skip the "solid ..." header line; chunk 0 starts at the first facet.
  const uint8_t *body_start = find_next_facet(base, end);
  if (body_start >= end) {
    *vertp = nullptr;
    *trip = nullptr;
    *nvertp = 0;
    *ntrip = 0;
    return 0;
  }

  // Partition [body_start, end) into nthreads chunks, aligning each
  // split point to the next "facet " line so that no facet straddles
  // a chunk boundary.
  std::vector<const uint8_t *> bounds(nthreads + 1);
  bounds[0] = body_start;
  bounds[nthreads] = end;
  size_t body_len = end - body_start;
  for (unsigned t = 1; t < nthreads; ++t) {
    const uint8_t *target = body_start + (body_len * t) / nthreads;
    bounds[t] = find_next_facet(target, end);
  }

  // A minimal facet is ~80 bytes; we use 60 as a conservative lower
  // bound to upper-bound the triangle count for tightly packed files.
  triangle_t ntris_max = (triangle_t)((body_len / 60) + 16);
  vertex_t nverts_max = 3 * ntris_max;

  // Concurrent hashtable. Capacity must hold every unique vertex with
  // load factor at most 50%. Saturate at the largest representable
  // power of two if the upper bound would overflow uint32.
  uint64_t vht_target = (uint64_t)nverts_max * 2 + 1;
  vertex_t vhtcap;
  if (vht_target >= (1ull << 31))
    vhtcap = 1u << 31;
  else
    vhtcap = nextpow2((uint32_t)vht_target);
  if (vhtcap < 1024)
    vhtcap = 1024;

  uint32_t *verts =
      (uint32_t *)malloc((size_t)nverts_max * 3 * sizeof(uint32_t));
  void *vht_backing = nullptr;
  size_t vht_backing_len = 0;
  std::atomic<uint32_t> *vht =
      alloc_atomic_table(vhtcap, vht_backing, vht_backing_len);
  if (!verts || !vht) {
    free(verts);
    free_atomic_table(vht, vht_backing, vht_backing_len);
    return -1;
  }
  std::atomic<uint32_t> nverts_atomic{0};

  std::vector<vertex_t *> chunk_tris(nthreads, nullptr);
  std::vector<triangle_t> chunk_ntris(nthreads, 0);
  std::atomic<int> error_flag{0};

  auto worker = [&](unsigned t) {
    const uint8_t *cs = bounds[t];
    const uint8_t *ce = bounds[t + 1];
    if (cs >= ce)
      return;
    triangle_t cap = (triangle_t)((size_t)(ce - cs) / 60 + 16);
    vertex_t *tris = (vertex_t *)malloc((size_t)cap * 3 * sizeof(vertex_t));
    if (!tris) {
      error_flag.store(1);
      return;
    }
    triangle_t n =
        parse_ascii_chunk(cs, ce, verts, vht, vhtcap, nverts_atomic, tris, cap);
    if (n == ~(triangle_t)0) {
      free(tris);
      error_flag.store(1);
      return;
    }
    chunk_tris[t] = tris;
    chunk_ntris[t] = n;
  };

  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  for (unsigned t = 0; t < nthreads; ++t)
    threads.emplace_back(worker, t);
  for (auto &th : threads)
    th.join();

  free_atomic_table(vht, vht_backing, vht_backing_len);

  if (error_flag.load()) {
    free(verts);
    for (auto p : chunk_tris)
      free(p);
    return -1;
  }

  // Concatenate chunk triangle arrays.
  size_t total_ntris = 0;
  for (auto n : chunk_ntris)
    total_ntris += n;
  vertex_t *tris = (vertex_t *)malloc(total_ntris * 3 * sizeof(vertex_t));
  if (!tris) {
    free(verts);
    for (auto p : chunk_tris)
      free(p);
    return -1;
  }
  size_t off = 0;
  for (unsigned t = 0; t < nthreads; ++t) {
    size_t n = chunk_ntris[t];
    if (n) {
      std::memcpy(tris + 3 * off, chunk_tris[t], n * 3 * sizeof(vertex_t));
      off += n;
    }
    free(chunk_tris[t]);
  }

  vertex_t nverts = nverts_atomic.load();
  verts = try_shrink(verts, (size_t)nverts * 3);
  *vertp = (float *)verts;
  *nvertp = nverts;
  *trip = tris;
  *ntrip = (triangle_t)total_ntris;
  return 0;
}

// ---------------------------------------------------------------------------
// Top-level dispatch.
// ---------------------------------------------------------------------------

// Triangle count above which the multi-threaded binary path is used.
static constexpr triangle_t MT_BINARY_THRESHOLD = 100000;
// File-size threshold above which the multi-threaded ASCII path is
// used. Roughly corresponds to >50k facets.
static constexpr size_t MT_ASCII_THRESHOLD = 4 * 1024 * 1024;
// Hard cap on the worker count to avoid pathological oversubscription
// on machines with very high logical core counts.
static constexpr unsigned MT_THREAD_CAP = 32;

// Resolve the worker thread count: ``PYVISTA_STL_THREADS`` if set,
// otherwise hardware_concurrency, capped at MT_THREAD_CAP. Returns at
// least 1.
static unsigned resolve_thread_count() {
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0)
    hw = 1;
  if (const char *env = std::getenv("PYVISTA_STL_THREADS")) {
    long v = std::strtol(env, nullptr, 10);
    if (v > 0)
      hw = (unsigned)v;
  }
  if (hw > MT_THREAD_CAP)
    hw = MT_THREAD_CAP;
  return hw;
}

// Return codes from loadstl_dispatch.
//   0  - success
//  -1  - I/O or allocation error
//  -2  - invalid or unrecognized format
//  -3  - declared triangle count exceeds the configured cap

static int loadstl_dispatch(const uint8_t *data, size_t size, float **vertp,
                            vertex_t *nvertp, vertex_t **trip,
                            triangle_t *ntrip) {
  StlFormat s = detect_format(data, size);
  if (s == STL_INVALID)
    return -2;
  unsigned nthreads = resolve_thread_count();
  triangle_t max_tris = resolve_max_tris();

  if (s == STL_BINARY) {
    uint32_t ntris;
    std::memcpy(&ntris, data + 80, sizeof(ntris));
    if (ntris > max_tris)
      return -3;
    if (ntris == 0) {
      *vertp = nullptr;
      *trip = nullptr;
      *nvertp = 0;
      *ntrip = 0;
      return 0;
    }
    if (nthreads > 1 && ntris >= MT_BINARY_THRESHOLD) {
      int rc = loadstl_binary_mt(data, size, ntris, nthreads, vertp, nvertp,
                                 trip, ntrip);
      if (rc == 0)
        return 0;
      // The MT path returned non-zero (typically because a degenerate
      // all-unique-vertex input saturated its smaller hashtable). Fall
      // through to the sequential path, which uses the worst-case
      // hashtable size.
    }
    return loadstl_binary_seq(data, size, ntris, vertp, nvertp, trip, ntrip);
  }

  if (nthreads > 1 && size >= MT_ASCII_THRESHOLD) {
    int rc = loadstl_ascii_mt(data, size, nthreads, vertp, nvertp, trip, ntrip);
    if (rc == 0)
      return 0;
    // The MT ASCII path can fail when a chunk's triangle estimate is
    // exceeded by an unusually compact writer. Fall back to the
    // sequential path which grows its buffers dynamically.
  }
  return loadstl_ascii(data, size, vertp, nvertp, trip, ntrip);
}

// ---------------------------------------------------------------------------
// nanobind binding
// ---------------------------------------------------------------------------

// Wrap a malloc-allocated buffer in a NumPy ndarray. The capsule
// deleter calls free so it matches the malloc/realloc allocator used
// by the parser. nanobind rejects null data even on zero-element
// dimensions, so for empty results we substitute a one-element
// allocation that the user-facing array still reports as empty.
template <typename T, size_t N>
static NDArray<T, N> wrap_malloc_buffer(T *src, std::array<int, N> shape) {
  size_t total = 1;
  for (size_t i = 0; i < N; ++i)
    total *= (size_t)shape[i];
  if (total == 0 || src == nullptr) {
    free(src);
    src = (T *)malloc(sizeof(T));
    if (!src)
      throw std::bad_alloc();
  }
  size_t shape_[N];
  for (size_t i = 0; i < N; ++i)
    shape_[i] = shape[i];
  nb::capsule owner(src, [](void *p) noexcept { std::free(p); });
  return NDArray<T, N>(src, N, shape_, owner);
}

nb::tuple GetStlData(const std::string &filename) {
  // Reject embedded NUL bytes: POSIX open(2) and Windows CreateFile
  // truncate at the first NUL, which can mask the user's intent and
  // hide path-confusion bugs in callers.
  if (filename.find('\0') != std::string::npos)
    throw std::invalid_argument("STL filename contains a NUL byte.");

  MappedFile mf;
  if (!mf.open(filename.c_str()))
    throw std::runtime_error("File not found or unreadable: " + filename);

  float *vertp = nullptr;
  unsigned int nverts = 0;
  unsigned int *trip = nullptr;
  unsigned int ntrip = 0;

  int rc;
  {
    nb::gil_scoped_release rel;
    rc = loadstl_dispatch(mf.data, mf.size, &vertp, &nverts, &trip, &ntrip);
  }

  if (rc != 0) {
    free(vertp);
    free(trip);
    if (rc == -2)
      throw std::runtime_error("Invalid or unrecognized STL file format.");
    if (rc == -3)
      throw std::runtime_error(
          "STL declares more triangles than PYVISTA_STL_MAX_TRIS allows.");
    throw std::runtime_error("Failed to load STL file.");
  }

  NDArray<float, 2> vert_arr =
      wrap_malloc_buffer<float, 2>(vertp, {(int)nverts, 3});
  NDArray<unsigned int, 2> face_arr =
      wrap_malloc_buffer<unsigned int, 2>(trip, {(int)ntrip, 3});

  return nb::make_tuple(vert_arr, face_arr);
}

NB_MODULE(_core, m) { m.def("get_stl_data", &GetStlData); }
