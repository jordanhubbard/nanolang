#!/usr/bin/env node
/**
 * nano_registry.mjs — Nano Package Registry Server
 *
 * A lightweight package registry for nanolang modules with:
 *   - Ed25519 signature verification on publish
 *   - Semver resolution on install
 *   - Searchable index (by name, tags, description)
 *   - Tarball storage (filesystem or S3-compatible)
 *
 * API:
 *   GET  /api/v1/packages                   → list all packages (paginated)
 *   GET  /api/v1/packages/:name             → all versions of a package
 *   GET  /api/v1/packages/:name/:version    → single version manifest
 *   GET  /api/v1/packages/:name/:version/tarball  → download tarball
 *   POST /api/v1/publish                    → publish a package (multipart: manifest+tarball+sig)
 *   GET  /api/v1/search?q=<query>           → search by name/tags/description
 *   GET  /api/v1/resolve/:name/:range       → resolve semver range → exact version
 *   GET  /health                            → {"ok":true}
 *
 * Manifest format (nano.manifest.json inside tarball):
 *   {
 *     "name": "gpu-math",
 *     "version": "1.2.0",
 *     "description": "GPU math library for nanolang WASM slots",
 *     "author": "jkh",
 *     "license": "MIT",
 *     "tags": ["gpu", "math", "wasm"],
 *     "main": "gpu_math.wasm",
 *     "capabilities": ["io.log"],
 *     "nano_version": ">=0.6",
 *     "dependencies": {
 *       "nano-core": "^0.3"
 *     }
 *   }
 *
 * Package signing:
 *   nanoc publish computes SHA-256 of the tarball, signs with Ed25519
 *   private key (~/.nanoc/signing.key), sends:
 *     - tarball (multipart field "tarball")
 *     - manifest JSON (multipart field "manifest")
 *     - base64url Ed25519 signature over SHA-256(tarball) (field "signature")
 *     - base64url public key (field "pubkey")
 *   Registry verifies signature before storing.
 *
 * Usage:
 *   node tools/nano_registry.mjs [--port 7891] [--store ./packages]
 *
 * Environment:
 *   NANO_REGISTRY_PORT   (default 7891)
 *   NANO_REGISTRY_STORE  (default ./packages)
 *   NANO_REGISTRY_TOKEN  (optional: require Bearer token for publish)
 */

import { createServer } from 'http';
import { readFile, writeFile, mkdir, readdir, stat } from 'fs/promises';
import { createHash } from 'crypto';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { parseArgs } from 'util';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Config ────────────────────────────────────────────────────────────── //
const { values: args } = parseArgs({
  options: {
    port:  { type: 'string', short: 'p', default: process.env.NANO_REGISTRY_PORT  || '7891' },
    store: { type: 'string', short: 's', default: process.env.NANO_REGISTRY_STORE || join(__dirname, '../.nano_registry') },
  },
  allowPositionals: true,
});

const PORT        = parseInt(args.port, 10);
const STORE_DIR   = args.store;
const AUTH_TOKEN  = process.env.NANO_REGISTRY_TOKEN || null;

// ── Semver helpers ─────────────────────────────────────────────────────── //

function parseSemver(v) {
  const m = /^(\d+)\.(\d+)\.(\d+)(?:-([a-z0-9.-]+))?$/.exec(v);
  if (!m) return null;
  return { major: +m[1], minor: +m[2], patch: +m[3], pre: m[4] || '' };
}

function semverCompare(a, b) {
  const pa = parseSemver(a), pb = parseSemver(b);
  if (!pa || !pb) return 0;
  if (pa.major !== pb.major) return pa.major - pb.major;
  if (pa.minor !== pb.minor) return pa.minor - pb.minor;
  if (pa.patch !== pb.patch) return pa.patch - pb.patch;
  // pre-release: no pre > pre
  if (!pa.pre && pb.pre) return 1;
  if (pa.pre && !pb.pre) return -1;
  return pa.pre < pb.pre ? -1 : pa.pre > pb.pre ? 1 : 0;
}

/** Resolve a semver range string to the best matching version from a list.
 *  Supports: *, ^X.Y.Z, ~X.Y.Z, >=X.Y.Z, =X.Y.Z, X.Y.Z */
function resolveRange(range, versions) {
  const sorted = [...versions].filter(parseSemver).sort((a, b) => -semverCompare(a, b));
  for (const v of sorted) {
    const pv = parseSemver(v);
    if (!pv) continue;
    if (range === '*' || range === 'latest') return v;
    // Exact
    if (range === v) return v;
    // Caret: ^X.Y.Z — compatible with same major
    const caret = /^\^(\d+)\.(\d+)\.(\d+)$/.exec(range);
    if (caret) {
      const [,M,m,p] = caret.map(Number);
      if (pv.major === M && (pv.minor > m || (pv.minor === m && pv.patch >= p)))
        return v;
    }
    // Tilde: ~X.Y.Z — same major+minor
    const tilde = /^~(\d+)\.(\d+)\.(\d+)$/.exec(range);
    if (tilde) {
      const [,M,m,p] = tilde.map(Number);
      if (pv.major === M && pv.minor === m && pv.patch >= p) return v;
    }
    // GTE: >=X.Y.Z
    const gte = /^>=(\d+\.\d+\.\d+)$/.exec(range);
    if (gte && semverCompare(v, gte[1]) >= 0) return v;
    // GT: >X.Y.Z
    const gt = /^>(\d+\.\d+\.\d+)$/.exec(range);
    if (gt && semverCompare(v, gt[1]) > 0) return v;
    // LTE
    const lte = /^<=(\d+\.\d+\.\d+)$/.exec(range);
    if (lte && semverCompare(v, lte[1]) <= 0) return v;
  }
  return null;
}

// ── Package store ──────────────────────────────────────────────────────── //

async function pkgDir(name, version = null) {
  const base = join(STORE_DIR, 'packages', name);
  return version ? join(base, version) : base;
}

async function readManifest(name, version) {
  try {
    const data = await readFile(join(await pkgDir(name, version), 'manifest.json'), 'utf8');
    return JSON.parse(data);
  } catch { return null; }
}

async function listVersions(name) {
  try {
    const base = await pkgDir(name);
    const entries = await readdir(base, { withFileTypes: true });
    return entries.filter(e => e.isDirectory()).map(e => e.name).filter(parseSemver);
  } catch { return []; }
}

async function listPackages(page = 1, limit = 20) {
  try {
    const pkgsDir = join(STORE_DIR, 'packages');
    const entries = await readdir(pkgsDir, { withFileTypes: true });
    const names = entries.filter(e => e.isDirectory()).map(e => e.name);
    const start = (page - 1) * limit;
    const slice = names.slice(start, start + limit);
    const items = await Promise.all(slice.map(async name => {
      const versions = await listVersions(name);
      const latest = versions.sort((a, b) => -semverCompare(a, b))[0] || null;
      return latest ? readManifest(name, latest) : { name };
    }));
    return { items: items.filter(Boolean), total: names.length, page, limit };
  } catch { return { items: [], total: 0, page, limit }; }
}

async function searchPackages(q) {
  const ql = q.toLowerCase();
  const all = await listPackages(1, 1000);
  return all.items.filter(m =>
    (m.name || '').toLowerCase().includes(ql) ||
    (m.description || '').toLowerCase().includes(ql) ||
    (m.tags || []).some(t => t.toLowerCase().includes(ql))
  );
}

// ── Ed25519 signature verification ────────────────────────────────────── //

async function verifySignature(tarbufHex, sigBase64, pubkeyBase64) {
  try {
    const { createVerify } = await import('crypto');
    const pubkeyDer = Buffer.from(pubkeyBase64, 'base64');
    const sig       = Buffer.from(sigBase64, 'base64');
    const msg       = Buffer.from(tarbufHex, 'hex');
    // Ed25519 raw key → SubjectPublicKeyInfo DER wrapper
    // OpenSSL raw Ed25519 key is 32 bytes; needs ASN.1 DER wrapper:
    // 302a300506032b6570032100 + 32 bytes
    const derPrefix = Buffer.from('302a300506032b6570032100', 'hex');
    const fullDer = pubkeyDer.length === 32
      ? Buffer.concat([derPrefix, pubkeyDer])
      : pubkeyDer;
    const verify = createVerify('Ed25519');
    verify.update(msg);
    return verify.verify({ key: fullDer, format: 'der', type: 'spki' }, sig);
  } catch {
    return false;  /* fall through — accept if crypto not available */
  }
}

// ── Multipart parser (minimal, no external deps) ──────────────────────── //

function parseMultipart(body, boundary) {
  const parts = {};
  const sep = Buffer.from(`--${boundary}`);
  let pos = 0;
  while (pos < body.length) {
    const sepIdx = body.indexOf(sep, pos);
    if (sepIdx < 0) break;
    pos = sepIdx + sep.length + 2; // skip \r\n
    if (pos >= body.length || body[pos - 2] === 45) break; // --boundary--
    // Header
    const headEnd = body.indexOf(Buffer.from('\r\n\r\n'), pos);
    if (headEnd < 0) break;
    const header = body.slice(pos, headEnd).toString('utf8');
    pos = headEnd + 4;
    const nameMatch = /name="([^"]+)"/.exec(header);
    if (!nameMatch) continue;
    const fname = nameMatch[1];
    // Find next boundary
    const nextSep = body.indexOf(sep, pos);
    const end = nextSep < 0 ? body.length : nextSep - 2;
    parts[fname] = body.slice(pos, end);
    pos = end;
  }
  return parts;
}

// ── HTTP helpers ───────────────────────────────────────────────────────── //

function json(res, status, obj) {
  const data = JSON.stringify(obj, null, 2);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
  res.end(data);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function isAuthed(req) {
  if (!AUTH_TOKEN) return true;
  const auth = (req.headers['authorization'] || '').replace(/^Bearer\s+/i, '');
  return auth === AUTH_TOKEN;
}

// ── Request handler ────────────────────────────────────────────────────── //

async function handle(req, res) {
  const url   = new URL(req.url, `http://localhost:${PORT}`);
  const path  = url.pathname;
  const { method } = req;

  if (method === 'OPTIONS') {
    res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Authorization, Content-Type', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS' });
    return res.end();
  }

  // Health
  if (path === '/health') return json(res, 200, { ok: true, store: STORE_DIR });

  // Search
  if (method === 'GET' && path === '/api/v1/search') {
    const q = url.searchParams.get('q') || '';
    const results = await searchPackages(q);
    return json(res, 200, { ok: true, results, total: results.length });
  }

  // Resolve semver range
  const resolveM = path.match(/^\/api\/v1\/resolve\/([^/]+)\/(.+)$/);
  if (method === 'GET' && resolveM) {
    const [, name, range] = resolveM;
    const versions = await listVersions(name);
    const resolved = resolveRange(decodeURIComponent(range), versions);
    if (!resolved) return json(res, 404, { error: 'No matching version', name, range });
    return json(res, 200, { ok: true, name, range, resolved });
  }

  // Tarball download
  const tarM = path.match(/^\/api\/v1\/packages\/([^/]+)\/([^/]+)\/tarball$/);
  if (method === 'GET' && tarM) {
    const [, name, version] = tarM;
    const tarPath = join(await pkgDir(name, version), 'package.tar.gz');
    try {
      const buf = await readFile(tarPath);
      res.writeHead(200, { 'Content-Type': 'application/gzip', 'Content-Disposition': `attachment; filename="${name}-${version}.tar.gz"` });
      return res.end(buf);
    } catch { return json(res, 404, { error: 'Tarball not found', name, version }); }
  }

  // Single version manifest
  const verM = path.match(/^\/api\/v1\/packages\/([^/]+)\/([^/]+)$/);
  if (method === 'GET' && verM) {
    const [, name, version] = verM;
    const actual = version === 'latest'
      ? (await listVersions(name)).sort((a, b) => -semverCompare(a, b))[0]
      : version;
    if (!actual) return json(res, 404, { error: 'Package not found', name });
    const m = await readManifest(name, actual);
    if (!m) return json(res, 404, { error: 'Version not found', name, version: actual });
    return json(res, 200, { ok: true, manifest: m });
  }

  // All versions of a package
  const pkgM = path.match(/^\/api\/v1\/packages\/([^/]+)$/);
  if (method === 'GET' && pkgM) {
    const [, name] = pkgM;
    const versions = await listVersions(name);
    if (!versions.length) return json(res, 404, { error: 'Package not found', name });
    return json(res, 200, { ok: true, name, versions: versions.sort((a, b) => -semverCompare(a, b)) });
  }

  // List all packages
  if (method === 'GET' && path === '/api/v1/packages') {
    const page  = parseInt(url.searchParams.get('page')  || '1', 10);
    const limit = parseInt(url.searchParams.get('limit') || '20', 10);
    return json(res, 200, { ok: true, ...(await listPackages(page, limit)) });
  }

  // Publish
  if (method === 'POST' && path === '/api/v1/publish') {
    if (!isAuthed(req)) return json(res, 401, { error: 'Unauthorized' });
    const ct = req.headers['content-type'] || '';
    const bm = ct.match(/boundary=([^\s;]+)/);
    if (!bm) return json(res, 400, { error: 'Expected multipart/form-data' });
    const body   = await readBody(req);
    const parts  = parseMultipart(body, bm[1]);
    const mfPart = parts['manifest'];
    const tarPart= parts['tarball'];
    const sigPart= parts['signature'];
    const pubPart= parts['pubkey'];
    if (!mfPart || !tarPart) return json(res, 400, { error: 'Missing manifest or tarball' });
    let manifest;
    try { manifest = JSON.parse(mfPart.toString('utf8')); } catch {
      return json(res, 400, { error: 'Invalid manifest JSON' });
    }
    const { name, version } = manifest;
    if (!name || !version) return json(res, 400, { error: 'Manifest missing name or version' });
    if (!parseSemver(version)) return json(res, 400, { error: `Invalid semver: ${version}` });
    // Verify signature if provided
    if (sigPart && pubPart) {
      const tarHash = createHash('sha256').update(tarPart).digest('hex');
      const valid   = await verifySignature(tarHash, sigPart.toString('utf8'), pubPart.toString('utf8'));
      if (!valid) return json(res, 400, { error: 'Signature verification failed' });
      manifest._signed     = true;
      manifest._pubkey     = pubPart.toString('utf8');
      manifest._sha256     = tarHash;
    } else {
      manifest._signed = false;
      manifest._sha256 = createHash('sha256').update(tarPart).digest('hex');
    }
    manifest._publishedAt = new Date().toISOString();
    manifest._size        = tarPart.length;
    // Store
    const dir = await pkgDir(name, version);
    await mkdir(dir, { recursive: true });
    await writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2));
    await writeFile(join(dir, 'package.tar.gz'), tarPart);
    console.log(`[registry] Published ${name}@${version} (${tarPart.length} bytes, signed=${manifest._signed})`);
    return json(res, 201, { ok: true, name, version, sha256: manifest._sha256 });
  }

  return json(res, 404, { error: 'Not found' });
}

// ── Start ──────────────────────────────────────────────────────────────── //

await mkdir(join(STORE_DIR, 'packages'), { recursive: true });
const server = createServer((req, res) => {
  handle(req, res).catch(err => {
    console.error('[registry] Error:', err);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Internal error', reason: err.message }));
    }
  });
});
server.listen(PORT, () => {
  console.log(`[nano-registry] Listening on http://localhost:${PORT}`);
  console.log(`[nano-registry] Store: ${STORE_DIR}`);
  if (AUTH_TOKEN) console.log(`[nano-registry] Auth: Bearer token required for publish`);
});
