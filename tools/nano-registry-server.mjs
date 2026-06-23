#!/usr/bin/env node
/**
 * nano-registry-server.mjs — nanolang package registry HTTP server
 *
 * A lightweight searchable index of published nano modules with semver resolution.
 * Stores packages as tarballs in a local store (or MinIO-compatible S3).
 *
 * Endpoints:
 *   GET  /packages                        — list all packages (name+latest version)
 *   GET  /packages/:name                  — package metadata + all versions
 *   GET  /packages/:name/:version         — specific version manifest
 *   GET  /packages/:name/:version/tarball — download tarball
 *   POST /packages                        — publish a package (signed tarball upload)
 *   GET  /search?q=<query>               — full-text search (name, description, tags)
 *   GET  /health                          — health check
 *
 * Auth:
 *   Publishing requires Authorization: Bearer <REGISTRY_TOKEN> header.
 *   Reading is public.
 *
 * Package manifest (nano.pkg.json inside tarball):
 *   {
 *     "name": "string",
 *     "version": "M.N.P",
 *     "description": "string",
 *     "author": "string",
 *     "license": "string",
 *     "tags": ["string"],
 *     "deps": { "<pkg>": "<semver_range>" },
 *     "main": "string",              // entry .nano file
 *     "ed25519_pubkey": "hex",       // publisher's public key (64 hex chars)
 *     "ed25519_sig": "hex"           // signature over tarball SHA-256 (128 hex chars)
 *   }
 *
 * Usage:
 *   REGISTRY_TOKEN=mysecret STORAGE_DIR=/var/nano-registry node nano-registry-server.mjs
 *   REGISTRY_TOKEN=mysecret PORT=3900 node nano-registry-server.mjs
 */

import { createServer } from 'http';
import { createReadStream, createWriteStream, promises as fs } from 'fs';
import { join, basename } from 'path';
import { pipeline } from 'stream/promises';
import { createHash } from 'crypto';
import { tmpdir } from 'os';
import { randomBytes } from 'crypto';

const PORT         = parseInt(process.env.PORT         || '3900');
const STORAGE_DIR  = process.env.STORAGE_DIR           || join(process.cwd(), 'nano-registry-store');
const TOKEN        = process.env.REGISTRY_TOKEN        || '';
const MAX_BODY     = parseInt(process.env.MAX_TARBALL_BYTES || String(20 * 1024 * 1024)); // 20 MB

// ─── Storage helpers ──────────────────────────────────────────────────────────

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true });
}

/** Returns path to directory for a package+version */
function pkgDir(name, version) {
  return join(STORAGE_DIR, 'pkgs', name, version);
}

/** Returns the index JSON path */
function indexPath() {
  return join(STORAGE_DIR, 'index.json');
}

/** Load or create the master index */
async function loadIndex() {
  try {
    const raw = await fs.readFile(indexPath(), 'utf8');
    return JSON.parse(raw);
  } catch {
    return {};  // { "<name>": { "latest": "version", "versions": { "M.N.P": manifest } } }
  }
}

async function saveIndex(idx) {
  await ensureDir(STORAGE_DIR);
  await fs.writeFile(indexPath(), JSON.stringify(idx, null, 2), 'utf8');
}

// ─── Semver helpers ───────────────────────────────────────────────────────────

function parseSemver(v) {
  const m = String(v).match(/^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$/);
  if (!m) return null;
  return { major: +m[1], minor: +m[2], patch: +m[3], pre: m[4] || '' };
}

function cmpSemver(a, b) {
  for (const k of ['major', 'minor', 'patch']) {
    if (a[k] !== b[k]) return a[k] - b[k];
  }
  // pre-release < release
  if (a.pre && !b.pre) return -1;
  if (!a.pre && b.pre) return 1;
  return a.pre < b.pre ? -1 : a.pre > b.pre ? 1 : 0;
}

/** Find best matching version from available list given a semver range.
 *  Supports: "1.2.3", "^1.2.3", "~1.2.3", ">=1.0.0", "*", "latest" */
function resolveRange(available, range) {
  if (!range || range === '*' || range === 'latest') {
    return available.sort((a, b) => {
      const pa = parseSemver(a), pb = parseSemver(b);
      return pa && pb ? cmpSemver(pb, pa) : 0;
    })[0] || null;
  }

  const pa = available.map(v => ({ v, p: parseSemver(v) })).filter(x => x.p);
  let candidates = [];

  const exact = parseSemver(range);
  if (exact) {
    candidates = pa.filter(x =>
      x.p.major === exact.major &&
      x.p.minor === exact.minor &&
      x.p.patch === exact.patch
    );
  } else if (range.startsWith('^')) {
    const base = parseSemver(range.slice(1));
    if (base) {
      candidates = pa.filter(x =>
        x.p.major === base.major &&
        cmpSemver(x.p, base) >= 0
      );
    }
  } else if (range.startsWith('~')) {
    const base = parseSemver(range.slice(1));
    if (base) {
      candidates = pa.filter(x =>
        x.p.major === base.major &&
        x.p.minor === base.minor &&
        cmpSemver(x.p, base) >= 0
      );
    }
  } else if (range.startsWith('>=')) {
    const base = parseSemver(range.slice(2).trim());
    if (base) candidates = pa.filter(x => cmpSemver(x.p, base) >= 0);
  } else if (range.startsWith('>')) {
    const base = parseSemver(range.slice(1).trim());
    if (base) candidates = pa.filter(x => cmpSemver(x.p, base) > 0);
  }

  if (!candidates.length) return null;
  candidates.sort((a, b) => cmpSemver(b.p, a.p));
  return candidates[0].v;
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

function json(res, status, obj) {
  const body = JSON.stringify(obj, null, 2);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

function checkAuth(req, res) {
  if (!TOKEN) return true;  // No token configured → open
  const h = req.headers['authorization'] || '';
  if (h.startsWith('Bearer ') && h.slice(7) === TOKEN) return true;
  json(res, 401, { error: 'Unauthorized — provide Authorization: Bearer <token>' });
  return false;
}

async function readBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', chunk => {
      size += chunk.length;
      if (size > maxBytes) {
        req.destroy(new Error(`Request body too large (> ${maxBytes} bytes)`));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

// ─── Route handlers ───────────────────────────────────────────────────────────

async function handleList(req, res) {
  const idx = await loadIndex();
  const pkgs = Object.entries(idx).map(([name, info]) => ({
    name,
    latest: info.latest,
    description: info.versions[info.latest]?.description || '',
    tags: info.versions[info.latest]?.tags || [],
    author: info.versions[info.latest]?.author || '',
  }));
  json(res, 200, { count: pkgs.length, packages: pkgs });
}

async function handleSearch(req, res, query) {
  const q = (query || '').toLowerCase().trim();
  if (!q) return handleList(req, res);
  const idx = await loadIndex();
  const results = [];
  for (const [name, info] of Object.entries(idx)) {
    const manifest = info.versions[info.latest] || {};
    const searchable = [
      name,
      manifest.description || '',
      (manifest.tags || []).join(' '),
      manifest.author || '',
    ].join(' ').toLowerCase();
    if (searchable.includes(q)) {
      results.push({
        name,
        latest: info.latest,
        description: manifest.description || '',
        tags: manifest.tags || [],
        author: manifest.author || '',
        score: name.includes(q) ? 2 : 1,  // name match ranks higher
      });
    }
  }
  results.sort((a, b) => b.score - a.score);
  json(res, 200, { query: q, count: results.length, packages: results });
}

async function handlePackageInfo(req, res, name) {
  const idx = await loadIndex();
  if (!idx[name]) return json(res, 404, { error: `Package '${name}' not found` });
  json(res, 200, {
    name,
    latest: idx[name].latest,
    versions: Object.keys(idx[name].versions).sort().reverse(),
    manifests: idx[name].versions,
  });
}

async function handleVersionInfo(req, res, name, version) {
  const idx = await loadIndex();
  if (!idx[name]) return json(res, 404, { error: `Package '${name}' not found` });

  // URL-decode the version (handles %5E → ^, %7E → ~, etc.)
  let decodedVersion = version;
  try { decodedVersion = decodeURIComponent(version); } catch { /* keep as-is */ }

  // Support semver range resolution
  const available = Object.keys(idx[name].versions);
  let resolved = decodedVersion;
  if (!idx[name].versions[decodedVersion]) {
    resolved = resolveRange(available, decodedVersion);
    if (!resolved) return json(res, 404, { error: `No version matching '${version}' found for '${name}'` });
  }
  json(res, 200, { name, resolved_version: resolved, manifest: idx[name].versions[resolved] });
}

async function handleTarball(req, res, name, version) {
  const idx = await loadIndex();
  if (!idx[name]) return json(res, 404, { error: `Package '${name}' not found` });

  let decodedVersion = version;
  try { decodedVersion = decodeURIComponent(version); } catch { /* keep as-is */ }

  const available = Object.keys(idx[name].versions);
  let resolved = decodedVersion;
  if (!idx[name].versions[decodedVersion]) {
    resolved = resolveRange(available, decodedVersion);
    if (!resolved) return json(res, 404, { error: `No version matching '${version}'` });
  }

  const tarPath = join(pkgDir(name, resolved), `${name}-${resolved}.tar.gz`);
  try {
    const stat = await fs.stat(tarPath);
    res.writeHead(200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': stat.size,
      'Content-Disposition': `attachment; filename="${basename(tarPath)}"`,
      'X-Package-Name': name,
      'X-Package-Version': resolved,
      'Access-Control-Allow-Origin': '*',
    });
    await pipeline(createReadStream(tarPath), res);
  } catch {
    json(res, 404, { error: `Tarball for ${name}@${resolved} not found on server` });
  }
}

async function handlePublish(req, res) {
  if (!checkAuth(req, res)) return;

  // Read multipart or raw body (raw tarball + JSON manifest in query params)
  // Simple approach: POST JSON body with { manifest: {...}, tarball_b64: "..." }
  let body;
  try {
    body = await readBody(req, MAX_BODY);
  } catch (e) {
    return json(res, 413, { error: e.message });
  }

  let data;
  try {
    data = JSON.parse(body.toString('utf8'));
  } catch {
    return json(res, 400, { error: 'Body must be JSON: { manifest: {...}, tarball_b64: "..." }' });
  }

  const { manifest, tarball_b64 } = data;
  if (!manifest || !manifest.name || !manifest.version) {
    return json(res, 400, { error: 'manifest.name and manifest.version are required' });
  }
  if (!parseSemver(manifest.version)) {
    return json(res, 400, { error: `Invalid semver: '${manifest.version}'` });
  }
  if (!tarball_b64) {
    return json(res, 400, { error: 'tarball_b64 (base64-encoded tarball) is required' });
  }

  const tarball = Buffer.from(tarball_b64, 'base64');
  const sha256  = createHash('sha256').update(tarball).digest('hex');
  manifest.sha256 = sha256;
  manifest.published_at = new Date().toISOString();

  // Save tarball
  const dir = pkgDir(manifest.name, manifest.version);
  await ensureDir(dir);
  await fs.writeFile(join(dir, `${manifest.name}-${manifest.version}.tar.gz`), tarball);
  await fs.writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');

  // Update index
  const idx = await loadIndex();
  if (!idx[manifest.name]) idx[manifest.name] = { latest: manifest.version, versions: {} };
  idx[manifest.name].versions[manifest.version] = manifest;

  // Update latest if newer
  const current = parseSemver(idx[manifest.name].latest);
  const incoming = parseSemver(manifest.version);
  if (!current || (incoming && cmpSemver(incoming, current) > 0)) {
    idx[manifest.name].latest = manifest.version;
  }

  await saveIndex(idx);

  json(res, 200, {
    ok: true,
    name: manifest.name,
    version: manifest.version,
    sha256,
    latest: idx[manifest.name].latest,
  });
}

// ─── Request router ───────────────────────────────────────────────────────────

const server = createServer(async (req, res) => {
  const urlObj = new URL(req.url, `http://localhost:${PORT}`);
  const path   = urlObj.pathname.replace(/\/+$/, '') || '/';
  const method = req.method.toUpperCase();

  try {
    // CORS preflight
    if (method === 'OPTIONS') {
      res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,OPTIONS', 'Access-Control-Allow-Headers': 'Authorization,Content-Type' });
      return res.end();
    }

    // GET /health
    if (method === 'GET' && path === '/health') {
      return json(res, 200, { ok: true, service: 'nano-registry', port: PORT });
    }

    // GET /packages
    if (method === 'GET' && path === '/packages') {
      return await handleList(req, res);
    }

    // GET /search?q=<query>
    if (method === 'GET' && path === '/search') {
      return await handleSearch(req, res, urlObj.searchParams.get('q') || '');
    }

    // POST /packages — publish
    if (method === 'POST' && path === '/packages') {
      return await handlePublish(req, res);
    }

    // GET /packages/:name
    const pkgMatch = path.match(/^\/packages\/([a-zA-Z0-9_\-]+)$/);
    if (method === 'GET' && pkgMatch) {
      return await handlePackageInfo(req, res, pkgMatch[1]);
    }

    // GET /packages/:name/:version (resolve + tarball download)
    // Version segment may include semver range chars: ^, ~, >, =, ., +
    const verMatch = path.match(/^\/packages\/([a-zA-Z0-9_\-]+)\/([^/]+)$/);
    if (method === 'GET' && verMatch) {
      const version = decodeURIComponent(verMatch[2]);
      const wantTar = req.headers['accept'] === 'application/octet-stream' ||
                      urlObj.searchParams.has('tarball');
      if (wantTar) return await handleTarball(req, res, verMatch[1], version);
      return await handleVersionInfo(req, res, verMatch[1], version);
    }

    // GET /packages/:name/:version/tarball
    const tarMatch = path.match(/^\/packages\/([a-zA-Z0-9_\-]+)\/([^/]+)\/tarball$/);
    if (method === 'GET' && tarMatch) {
      return await handleTarball(req, res, tarMatch[1], tarMatch[2]);
    }

    json(res, 404, { error: `Not found: ${method} ${path}` });
  } catch (err) {
    console.error('[registry] Error:', err);
    json(res, 500, { error: 'Internal server error', detail: err.message });
  }
});

await ensureDir(STORAGE_DIR);
await ensureDir(join(STORAGE_DIR, 'pkgs'));

server.listen(PORT, () => {
  console.log(`[nano-registry] Listening on port ${PORT}`);
  console.log(`[nano-registry] Storage: ${STORAGE_DIR}`);
  console.log(`[nano-registry] Auth token: ${TOKEN ? 'configured' : 'NONE (open)'}`);
});

server.on('error', err => { console.error('[nano-registry] Fatal:', err); process.exit(1); });
