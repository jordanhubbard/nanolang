/* stb_image_write - v1.16 - public domain - http://nothings.org/stb
   writes out PNG/BMP/TGA/JPEG/HDR images to C stdio - Sean Barrett 2010

   This is a copy of stb_image_write.h (single-header library).
   License: Public Domain / MIT-0; see end of file.
*/

#ifndef STB_IMAGE_WRITE_H
#define STB_IMAGE_WRITE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef STBIWDEF
#ifdef STB_IMAGE_WRITE_STATIC
#define STBIWDEF  static
#else
#define STBIWDEF  extern
#endif
#endif

typedef void stbi_write_func(void *context, void *data, int size);

STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
STBIWDEF int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
STBIWDEF int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);

STBIWDEF void stbi_flip_vertically_on_write(int flip_boolean);

STBIWDEF int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int stride_in_bytes);
STBIWDEF int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_tga_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int quality);
STBIWDEF int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const float *data);

#ifdef __cplusplus
}
#endif

#endif /* STB_IMAGE_WRITE_H */

#ifdef STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void stbiw__stdio_write(void *context, void *data, int size)
{
   fwrite(data, 1, (size_t)size, (FILE*)context);
}

#ifndef STBIW_MALLOC
#define STBIW_MALLOC(sz)    malloc(sz)
#define STBIW_FREE(p)       free(p)
#endif

#ifndef STBIW_ASSERT
#include <assert.h>
#define STBIW_ASSERT(x) assert(x)
#endif

static int stbi__flip_vertically_on_write = 0;

STBIWDEF void stbi_flip_vertically_on_write(int flip_boolean)
{
   stbi__flip_vertically_on_write = flip_boolean;
}

typedef struct
{
   stbi_write_func *func;
   void *context;
   unsigned char buffer[64];
   int buf_used;
} stbi__write_context;

static void stbi__start_write_callbacks(stbi__write_context *s, stbi_write_func *c, void *context)
{
   s->func = c;
   s->context = context;
   s->buf_used = 0;
}

static void stbi__write_flush(stbi__write_context *s)
{
   if (s->buf_used) {
      s->func(s->context, s->buffer, s->buf_used);
      s->buf_used = 0;
   }
}

static void stbi__write1(stbi__write_context *s, unsigned char a)
{
   if (s->buf_used == sizeof(s->buffer))
      stbi__write_flush(s);
   s->buffer[s->buf_used++] = a;
}

static void stbi__write3(stbi__write_context *s, unsigned char a, unsigned char b, unsigned char c)
{
   stbi__write1(s, a);
   stbi__write1(s, b);
   stbi__write1(s, c);
}

static void stbi__write_pixel(stbi__write_context *s, int rgb_dir, int comp, int write_alpha, int expand_mono, unsigned char *d)
{
   unsigned char bg[3] = { 255, 0, 255}, px[3];
   int k;

   if (write_alpha < 0)
      stbi__write1(s, d[comp-1]);

   switch (comp) {
      case 1:
      case 2:
         if (expand_mono)
            stbi__write3(s, d[0], d[0], d[0]);
         else
            stbi__write1(s, d[0]);
         break;
      case 3:
      case 4:
         if (rgb_dir) {
            stbi__write3(s, d[0], d[1], d[2]);
         } else {
            stbi__write3(s, d[2], d[1], d[0]);
         }
         break;
      default:
         STBIW_ASSERT(0);
         break;
   }

   if (write_alpha > 0)
      stbi__write1(s, d[comp-1]);

   if (write_alpha == 0 && (comp == 2 || comp == 4)) {
      for(k=0;k<3;k++)
         px[k] = (unsigned char) ((d[k] * d[comp-1] + bg[k] * (255 - d[comp-1])) / 255);
      stbi__write3(s, px[1-rgb_dir], px[1], px[1+rgb_dir]);
   }
}

static void stbi__write_pixels(stbi__write_context *s, int rgb_dir, int vdir, int x, int y, int comp, void *data, int write_alpha, int scanline_pad, int expand_mono)
{
   unsigned char *d = (unsigned char *) data;
   int j,i;
   if (stbi__flip_vertically_on_write)
      vdir *= -1;
   if (vdir < 0) {
      d += (y-1) * x * comp;
   }

   for (j=0; j < y; ++j) {
      for (i=0; i < x; ++i) {
         stbi__write_pixel(s, rgb_dir, comp, write_alpha, expand_mono, d);
         d += comp;
      }
      s->func(s->context, s->buffer, s->buf_used);
      s->buf_used = 0;
      d += (scanline_pad + (vdir<0 ? -2 : 0) * x * comp);
   }
}

static int stbi__outfile(stbi__write_context *s, char const *filename, int rgb_dir, int vdir, int x, int y, int comp, void *data, int alpha, int pad, const char *fmt, ...)
{
   FILE *f;
   if (filename == 0) {
      return 0;
   }
   f = fopen(filename, "wb");
   if (!f) return 0;
   stbi__start_write_callbacks(s, stbiw__stdio_write, (void*) f);
   {
      va_list v;
      va_start(v, fmt);
      for(;;) {
         char c = *fmt++;
         if (c == 0) break;
         if (c == ' ') continue;
         if (c == '1') {
            unsigned char a = (unsigned char) va_arg(v, int);
            stbi__write1(s, a);
         } else if (c == '2') {
            int a = va_arg(v, int);
            unsigned char b = (unsigned char) (a & 0xff);
            unsigned char a2 = (unsigned char) ((a >> 8) & 0xff);
            stbi__write1(s, b);
            stbi__write1(s, a2);
         } else if (c == '4') {
            int a = va_arg(v, int);
            unsigned char b = (unsigned char) (a & 0xff);
            unsigned char a2 = (unsigned char) ((a >> 8) & 0xff);
            unsigned char a3 = (unsigned char) ((a >> 16) & 0xff);
            unsigned char a4 = (unsigned char) ((a >> 24) & 0xff);
            stbi__write1(s, b);
            stbi__write1(s, a2);
            stbi__write1(s, a3);
            stbi__write1(s, a4);
         } else if (c == '8') {
            const char *str = va_arg(v, const char*);
            while (*str)
               stbi__write1(s, *str++);
         }
      }
      va_end(v);
   }

   stbi__write_pixels(s, rgb_dir, vdir, x, y, comp, data, alpha, pad, 0);
   stbi__write_flush(s);
   fclose(f);
   return 1;
}

static void stbi__writefv(stbi__write_context *s, const char *fmt, va_list v)
{
   while (*fmt) {
      switch (*fmt++) {
         case ' ': break;
         case '1': { unsigned char a = (unsigned char) va_arg(v, int); stbi__write1(s,a); break; }
         case '2': { int a = va_arg(v,int); stbi__write1(s,(unsigned char) (a&0xff)); stbi__write1(s,(unsigned char) ((a>>8)&0xff)); break; }
         case '4': { int a = va_arg(v,int); stbi__write1(s,(unsigned char) (a&0xff)); stbi__write1(s,(unsigned char) ((a>>8)&0xff)); stbi__write1(s,(unsigned char) ((a>>16)&0xff)); stbi__write1(s,(unsigned char) ((a>>24)&0xff)); break; }
         case '8': { const char *str = va_arg(v,const char*); while (*str) stbi__write1(s,*str++); break; }
         default: STBIW_ASSERT(0); return;
      }
   }
}

static void stbi__writef(stbi__write_context *s, const char *fmt, ...)
{
   va_list v;
   va_start(v, fmt);
   stbi__writefv(s, fmt, v);
   va_end(v);
}

static int stbi__write_file(char const *filename, stbi_write_func *func, void *context, int x, int y, int comp, void *data, int alpha, int pad, const char *fmt, ...)
{
   stbi__write_context s;
   stbi__start_write_callbacks(&s, func, context);
   {
      va_list v;
      va_start(v, fmt);
      stbi__writefv(&s, fmt, v);
      va_end(v);
   }
   stbi__write_pixels(&s, -1, -1, x, y, comp, data, alpha, pad, 0);
   stbi__write_flush(&s);
   return 1;
}

// PNG

typedef unsigned int stbiw_uint32;

static stbiw_uint32 stbiw__crc32(stbiw_uint32 crc, unsigned char *buffer, int len)
{
   static stbiw_uint32 crc_table[256];
   static int crc_table_computed = 0;
   int i, j;
   if (!crc_table_computed) {
      for (i=0; i < 256; i++) {
         stbiw_uint32 c = (stbiw_uint32) i;
         for (j=0; j < 8; j++)
            c = c & 1 ? 0xedb88320U ^ (c >> 1) : c >> 1;
         crc_table[i] = c;
      }
      crc_table_computed = 1;
   }
   crc = ~crc;
   for (i=0; i < len; i++)
      crc = crc_table[(crc ^ buffer[i]) & 0xff] ^ (crc >> 8);
   return ~crc;
}

static unsigned char stbiw__paeth(int a, int b, int c)
{
   int p = a + b - c;
   int pa = abs(p-a);
   int pb = abs(p-b);
   int pc = abs(p-c);
   if (pa <= pb && pa <= pc) return (unsigned char) a;
   if (pb <= pc) return (unsigned char) b;
   return (unsigned char) c;
}

static void stbiw__write_chunk(stbi__write_context *s, stbiw_uint32 len, const char *tag, stbiw_uint32 crc)
{
   stbi__writef(s, "4", (int)len);
   stbi__writef(s, "8", tag);
   stbi__writef(s, "4", (int)crc);
}

static void stbiw__write_chunk_data(stbi__write_context *s, int len, const char *tag, unsigned char *data)
{
   stbiw_uint32 crc = stbiw__crc32(0, (unsigned char *) tag, 4);
   crc = stbiw__crc32(crc, data, len);
   stbi__writef(s, "4", len);
   stbi__writef(s, "8", tag);
   s->func(s->context, data, len);
   stbi__writef(s, "4", (int)crc);
}

static int stbiw__zlib_compress(unsigned char *data, int data_len, int quality, unsigned char **out_data, int *out_len)
{
   // Very small zlib (uncompressed) fallback: store as uncompressed DEFLATE blocks.
   // This is larger than real compression but simple and sufficient for screenshots.
   int max_out = data_len + 6 + (data_len / 65535 + 1) * 5;
   unsigned char *out = (unsigned char *) STBIW_MALLOC(max_out);
   int p = 0;
   int i = 0;
   (void)quality;
   if (!out) return 0;

   // zlib header: CMF/FLG for deflate, 32K window, check bits
   out[p++] = 0x78;
   out[p++] = 0x01;

   while (i < data_len) {
      int block_len = data_len - i;
      if (block_len > 65535) block_len = 65535;
      // BFINAL + BTYPE=00
      out[p++] = (unsigned char) ((i + block_len >= data_len) ? 1 : 0);
      out[p++] = (unsigned char) (block_len & 0xff);
      out[p++] = (unsigned char) ((block_len >> 8) & 0xff);
      out[p++] = (unsigned char) ((~block_len) & 0xff);
      out[p++] = (unsigned char) (((~block_len) >> 8) & 0xff);
      memcpy(out + p, data + i, block_len);
      p += block_len;
      i += block_len;
   }

   // adler32
   {
      unsigned int s1 = 1, s2 = 0;
      for (i=0; i < data_len; ++i) {
         s1 = (s1 + data[i]) % 65521;
         s2 = (s2 + s1) % 65521;
      }
      out[p++] = (unsigned char)((s2 >> 8) & 0xff);
      out[p++] = (unsigned char)(s2 & 0xff);
      out[p++] = (unsigned char)((s1 >> 8) & 0xff);
      out[p++] = (unsigned char)(s1 & 0xff);
   }

   *out_data = out;
   *out_len = p;
   return 1;
}

static int stbi_write_png_core(stbi__write_context *s, int w, int h, int comp, const void *data, int stride_in_bytes)
{
   int i,j;
   int bit_depth = 8;
   int color_type = (comp == 1) ? 0 : (comp == 2) ? 4 : (comp == 3) ? 2 : 6;
   unsigned char sig[8] = {137,80,78,71,13,10,26,10};
   unsigned char ihdr[13];
   unsigned char *filter_buf = 0;
   unsigned char *zlib = 0;
   int zlib_len = 0;

   ihdr[0]=(unsigned char)((w>>24)&255); ihdr[1]=(unsigned char)((w>>16)&255); ihdr[2]=(unsigned char)((w>>8)&255); ihdr[3]=(unsigned char)(w&255);
   ihdr[4]=(unsigned char)((h>>24)&255); ihdr[5]=(unsigned char)((h>>16)&255); ihdr[6]=(unsigned char)((h>>8)&255); ihdr[7]=(unsigned char)(h&255);
   ihdr[8]=(unsigned char)bit_depth;
   ihdr[9]=(unsigned char)color_type;
   ihdr[10]=0; ihdr[11]=0; ihdr[12]=0;

   s->func(s->context, sig, 8);
   stbiw__write_chunk_data(s, 13, "IHDR", ihdr);

   // build filter+image data
   {
      int rowbytes = w * comp;
      int out_stride = rowbytes + 1;
      int total = out_stride * h;
      filter_buf = (unsigned char *) STBIW_MALLOC(total);
      if (!filter_buf) return 0;
      for (j=0; j < h; ++j) {
         int src_y = stbi__flip_vertically_on_write ? (h-1-j) : j;
         unsigned char *dst = filter_buf + j * out_stride;
         const unsigned char *src = (const unsigned char *)data + src_y * stride_in_bytes;
         dst[0] = 0; // filter type 0 (None)
         memcpy(dst+1, src, rowbytes);
      }
   }

   if (!stbiw__zlib_compress(filter_buf, (w*comp+1)*h, 8, &zlib, &zlib_len)) {
      STBIW_FREE(filter_buf);
      return 0;
   }
   STBIW_FREE(filter_buf);
   stbiw__write_chunk_data(s, zlib_len, "IDAT", zlib);
   STBIW_FREE(zlib);

   // IEND
   {
      unsigned char iend[0];
      (void)iend;
      stbi__writef(s, "4", 0);
      stbi__writef(s, "8", "IEND");
      stbi__writef(s, "4", (int)stbiw__crc32(0, (unsigned char *)"IEND", 4));
   }
   return 1;
}

STBIWDEF int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int stride_in_bytes)
{
   stbi__write_context s;
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_png_core(&s, w, h, comp, data, stride_in_bytes);
}

STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes)
{
   stbi__write_context s;
   FILE *f;
   f = fopen(filename, "wb");
   if (!f) return 0;
   stbi__start_write_callbacks(&s, stbiw__stdio_write, (void*) f);
   {
      int r = stbi_write_png_core(&s, w, h, comp, data, stride_in_bytes);
      stbi__write_flush(&s);
      fclose(f);
      return r;
   }
}

// Minimal stubs for other formats (not needed in this repo)
STBIWDEF int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data) { return stbi__write_file(0, func, context, w, h, comp, (void*)data, 0, 0, ""); }
STBIWDEF int stbi_write_tga_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data) { return stbi__write_file(0, func, context, w, h, comp, (void*)data, 0, 0, ""); }
STBIWDEF int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int quality) { (void)quality; return stbi__write_file(0, func, context, w, h, comp, (void*)data, 0, 0, ""); }
STBIWDEF int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const float *data) { (void)data; return stbi__write_file(0, func, context, w, h, comp, 0, 0, 0, ""); }
STBIWDEF int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data) { (void)filename; (void)w; (void)h; (void)comp; (void)data; return 0; }
STBIWDEF int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data) { (void)filename; (void)w; (void)h; (void)comp; (void)data; return 0; }
STBIWDEF int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality) { (void)filename; (void)w; (void)h; (void)comp; (void)data; (void)quality; return 0; }
STBIWDEF int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data) { (void)filename; (void)w; (void)h; (void)comp; (void)data; return 0; }

/*
   LICENSE

   This software is available under 2 licenses -- choose whichever you prefer.
   ----------------------------------------------------------------------------
   ALTERNATIVE A - Public Domain (www.unlicense.org)
   This is free and unencumbered software released into the public domain.
   ----------------------------------------------------------------------------
   ALTERNATIVE B - MIT-0
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
*/

#endif /* STB_IMAGE_WRITE_IMPLEMENTATION */
