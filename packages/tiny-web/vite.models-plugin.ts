import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin, ViteDevServer } from "vite";
import {
  EXPORTS_DIR,
  listExports,
  readExportIndex,
  saveVersionedExport,
  type ModelExportMeta,
} from "../tiny-llm/src/model-store.ts";
import type { ModelManifest } from "../tiny-llm/src/model-io.ts";

function repoRootFromConfig(root: string): string {
  return path.resolve(root, "../..");
}

async function readRawBody(req: IncomingMessage, limitMb = 512): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  const max = limitMb * 1024 * 1024;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buf.length;
    if (size > max) throw new Error(`Body too large (>${limitMb} MB)`);
    chunks.push(buf);
  }
  return Buffer.concat(chunks);
}

function sendJson(res: ServerResponse, status: number, data: unknown): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(data));
}

function modelsApiPlugin(): Plugin {
  let baseDir = process.cwd();

  const attach = (server: ViteDevServer) => {
    baseDir = repoRootFromConfig(server.config.root);

    server.middlewares.use((req, res, next) => {
      const pathname = req.url?.split("?")[0] ?? "";
      if (!pathname.startsWith("/models/")) return next();
      const filePath = path.join(baseDir, pathname.slice(1));
      const modelsRoot = path.join(baseDir, "models");
      if (!filePath.startsWith(modelsRoot) || !fs.existsSync(filePath)) {
        next();
        return;
      }
      if (fs.statSync(filePath).isDirectory()) {
        next();
        return;
      }
      res.setHeader("Cache-Control", "no-store");
      fs.createReadStream(filePath).pipe(res);
    });

    server.middlewares.use(async (req, res, next) => {
      if (!req.url?.startsWith("/api/models")) return next();
      const url = new URL(req.url, "http://local");

      if (req.method === "GET" && url.pathname === "/api/models/list") {
        sendJson(res, 200, { exports: listExports(baseDir) });
        return;
      }

      if (req.method === "POST" && url.pathname === "/api/models/export") {
        try {
          const weightsBuf = await readRawBody(req);
          const meta = JSON.parse(
            req.headers["x-export-meta"] as string,
          ) as Omit<ModelExportMeta, "version">;
          const manifest = JSON.parse(
            req.headers["x-export-manifest"] as string,
          ) as ModelManifest;
          const weights = new Float32Array(
            weightsBuf.buffer,
            weightsBuf.byteOffset,
            weightsBuf.byteLength / 4,
          );
          const saved = saveVersionedExport(
            { manifest, weights, meta },
            baseDir,
          );
          sendJson(res, 200, saved);
        } catch (err) {
          sendJson(res, 500, {
            error: err instanceof Error ? err.message : String(err),
          });
        }
        return;
      }

      if (req.method === "GET" && url.pathname === "/api/models/index") {
        sendJson(res, 200, readExportIndex(path.join(baseDir, EXPORTS_DIR)));
        return;
      }

      next();
    });
  };

  return {
    name: "tiny-gpt-models",
    configureServer: attach,
    configurePreviewServer: attach,
  };
}

export default modelsApiPlugin;
