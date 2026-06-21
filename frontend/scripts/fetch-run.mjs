#!/usr/bin/env node
/**
 * fetch-run.mjs — One-shot fetch (Decision A).
 *
 * Reads Nebius credentials from the environment or an AWS profile, downloads a
 * single run's artifacts (robot.urdf, meshes/*.glb, trajectory.json) from a
 * Nebius Object Storage bucket into public/runs/<run_id>/, and updates
 * public/runs/index.json.
 *
 * SECURITY: credentials are resolved at runtime by the AWS SDK's default
 * provider chain (env vars or ~/.aws profile). They are never written to disk
 * by this script and never reach the browser — only the downloaded artifacts
 * are served statically.
 *
 * Usage:
 *   npm run fetch -- <run_id>
 *   node scripts/fetch-run.mjs <run_id>
 */

import { createWriteStream } from 'node:fs';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';

import 'dotenv/config';
import {
  S3Client,
  ListObjectsV2Command,
  GetObjectCommand,
} from '@aws-sdk/client-s3';
import { fromNodeProviderChain } from '@aws-sdk/credential-providers';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');
const RUNS_DIR = join(PROJECT_ROOT, 'public', 'runs');

function die(message) {
  console.error(`\n✗ ${message}\n`);
  process.exit(1);
}

// ---- 1. Validate inputs -----------------------------------------------------

const runId = process.argv[2];
if (!runId) {
  die(
    'Missing run id.\n  Usage: npm run fetch -- <run_id>\n' +
      '  Example: npm run fetch -- 2026-06-21_ppo_run_0042',
  );
}

const ENDPOINT = process.env.NEBIUS_S3_ENDPOINT;
const REGION = process.env.NEBIUS_REGION;
const BUCKET = process.env.NEBIUS_BUCKET;
// Prefix may legitimately be empty (runs at bucket root).
const RUN_PREFIX = process.env.NEBIUS_RUN_PREFIX ?? '';
const FORCE_PATH_STYLE =
  String(process.env.NEBIUS_S3_FORCE_PATH_STYLE).toLowerCase() === 'true';

const missing = [];
if (!ENDPOINT) missing.push('NEBIUS_S3_ENDPOINT');
if (!REGION) missing.push('NEBIUS_REGION');
if (!BUCKET) missing.push('NEBIUS_BUCKET');
if (missing.length) {
  die(
    `Missing required env var(s): ${missing.join(', ')}.\n` +
      '  Copy .env.example to .env and fill it in (see the README).',
  );
}

// Join prefix + run id into an S3 key prefix, normalising slashes.
const keyPrefix =
  [RUN_PREFIX.replace(/\/+$/, ''), runId.replace(/^\/+|\/+$/g, '')]
    .filter(Boolean)
    .join('/') + '/';

// ---- 2. Build the S3 client (Nebius is S3-compatible) -----------------------

// Resolve credentials explicitly so we can give a clear error if none are found,
// instead of an opaque SDK failure deep inside the first request.
let credentials;
try {
  credentials = await fromNodeProviderChain()();
} catch (err) {
  die(
    'Could not resolve Nebius credentials.\n' +
      '  Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in .env, or set\n' +
      '  AWS_PROFILE=nebius with a matching ~/.aws/credentials entry.\n' +
      `  (underlying error: ${err.message})`,
  );
}

const s3 = new S3Client({
  endpoint: ENDPOINT,
  region: REGION,
  forcePathStyle: FORCE_PATH_STYLE,
  credentials,
});

// ---- 3. List + download -----------------------------------------------------

async function listKeys() {
  const keys = [];
  let ContinuationToken;
  do {
    const out = await s3.send(
      new ListObjectsV2Command({
        Bucket: BUCKET,
        Prefix: keyPrefix,
        ContinuationToken,
      }),
    );
    for (const obj of out.Contents ?? []) {
      if (obj.Key && !obj.Key.endsWith('/')) keys.push(obj.Key);
    }
    ContinuationToken = out.IsTruncated ? out.NextContinuationToken : undefined;
  } while (ContinuationToken);
  return keys;
}

async function downloadKey(key) {
  // Strip the run prefix so files land at public/runs/<run_id>/<relative>.
  const relative = key.slice(keyPrefix.length);
  const destPath = join(RUNS_DIR, runId, relative);
  await mkdir(dirname(destPath), { recursive: true });

  const { Body } = await s3.send(
    new GetObjectCommand({ Bucket: BUCKET, Key: key }),
  );
  if (!Body) throw new Error(`empty body for ${key}`);
  await pipeline(Body, createWriteStream(destPath));
  return relative;
}

async function updateIndex(runIds) {
  const indexPath = join(RUNS_DIR, 'index.json');
  let existing = [];
  try {
    existing = JSON.parse(await readFile(indexPath, 'utf8')).runs ?? [];
  } catch {
    /* no index yet */
  }
  const runs = Array.from(new Set([...existing, ...runIds])).sort();
  await writeFile(indexPath, JSON.stringify({ runs }, null, 2) + '\n');
  return runs;
}

// ---- main -------------------------------------------------------------------

console.log(`→ Bucket   : ${BUCKET}`);
console.log(`→ Endpoint : ${ENDPOINT}  (path-style: ${FORCE_PATH_STYLE})`);
console.log(`→ Prefix   : ${keyPrefix}`);

let keys;
try {
  keys = await listKeys();
} catch (err) {
  die(
    `Failed to list objects under "${keyPrefix}".\n` +
      '  Check NEBIUS_BUCKET / NEBIUS_RUN_PREFIX / run_id and that your key has\n' +
      '  read access. If you see a TLS or DNS error, try\n' +
      '  NEBIUS_S3_FORCE_PATH_STYLE=true.\n' +
      `  (underlying error: ${err.name}: ${err.message})`,
  );
}

if (keys.length === 0) {
  die(
    `No objects found under "${keyPrefix}" in bucket "${BUCKET}".\n` +
      '  Verify the run_id and NEBIUS_RUN_PREFIX are correct.',
  );
}

console.log(`\nFound ${keys.length} object(s). Downloading…`);
const downloaded = [];
for (const key of keys) {
  const rel = await downloadKey(key);
  downloaded.push(rel);
  console.log(`  ✓ ${rel}`);
}

// Sanity-check the two files the viewer needs.
const hasUrdf = downloaded.some((r) => r.toLowerCase().endsWith('.urdf'));
const hasTraj = downloaded.includes('trajectory.json');
if (!hasUrdf) console.warn('  ⚠ no .urdf file in this run — the viewer needs one.');
if (!hasTraj) console.warn('  ⚠ no trajectory.json — playback will be empty.');

const runs = await updateIndex([runId]);
console.log(`\n✓ Run "${runId}" ready at public/runs/${runId}/`);
console.log(`✓ index.json now lists: ${runs.join(', ')}`);
console.log(`\nNext: npm run dev  →  open  http://localhost:5173/?run=${runId}\n`);
