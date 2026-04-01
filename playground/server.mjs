// nano eval server — wraps the `bin/nano` interpreter
import { createServer } from "http";
import { execFile } from "child_process";
import { writeFile, unlink, readFile } from "fs/promises";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const PORT = process.env.PORT || 8792;
const NANO_BIN = join(ROOT, "bin", "nano");
const INDEX_HTML = join(__dirname, "index.html");

async function evalSource(source, timeout = 5000) {
  const tmpFile = join(tmpdir(), `nano_${randomUUID()}.nano`);
  try {
    await writeFile(tmpFile, source, "utf8");
    return await new Promise((resolve) => {
      execFile(NANO_BIN, [tmpFile], { timeout, cwd: ROOT }, (err, stdout, stderr) => {
        if (err?.killed) {
          resolve({ output: "", error: "Timeout: program exceeded 5s limit" });
        } else {
          const errMsg = stderr || (err && !stdout ? err.message : "");
          resolve({ output: stdout, error: errMsg });
        }
      });
    });
  } finally {
    await unlink(tmpFile).catch(() => {});
  }
}

createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") { res.writeHead(204); return res.end(); }

  if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
    try {
      const html = await readFile(INDEX_HTML, "utf8");
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      return res.end(html);
    } catch { res.writeHead(500); return res.end("Could not read index.html"); }
  }

  if (req.method === "POST" && req.url === "/eval") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", async () => {
      try {
        const { source = "", timeout = 5000 } = JSON.parse(body);
        const result = await evalSource(source, Math.min(Number(timeout), 10000));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ output: "", error: String(e) }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end("Not found");
}).listen(PORT, () => {
  console.log(`Nanolang Playground  →  http://localhost:${PORT}`);
  console.log(`Interpreter          →  ${NANO_BIN}`);
});
