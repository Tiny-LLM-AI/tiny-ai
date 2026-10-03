import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin, ViteDevServer } from "vite";
import {
  EXPORTS_DIR,
  listExports,
  listSavedModels,
  readExportIndex,
  saveModelToFolder,
  defaultSaveFolderName,
  saveVersionedExport,
  type ModelExportMeta,
} from "../tiny-llm/src/model-store.ts";
import type { ModelManifest } from "../tiny-llm/src/model-io.ts";
import { decodeSavePayload } from "./src/model-save-codec.ts";

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
      if (req.method === "GET" && pathname === "/api/corpus/vi-foundation") {
        const corpusPath = path.join(baseDir, "train-data/vi-foundation.txt");
        if (!fs.existsSync(corpusPath)) {
          sendJson(res, 404, { error: "Missing corpus. Run npm run fetch:vi-foundation." });
          return;
        }
        res.setHeader("Content-Type", "text/plain; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        fs.createReadStream(corpusPath).pipe(res);
        return;
      }
      if (req.method === "GET" && pathname === "/api/corpus/vi-chat-basic") {
        const corpusPath = path.join(baseDir, "train-data/vi-chat-basic.txt");
        if (!fs.existsSync(corpusPath)) { sendJson(res, 404, { error: "Missing chat corpus." }); return; }
        res.setHeader("Content-Type", "text/plain; charset=utf-8");
        res.setHeader("Cache-Control", "no-store");
        fs.createReadStream(corpusPath).pipe(res);
        return;
      }
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
        const saved = listSavedModels(baseDir);
        const legacy = listExports(baseDir);
        const seen = new Set(saved.map((e) => e.id));
        const merged = [
          ...saved,
          ...legacy.filter((e) => !seen.has(e.id)),
        ];
        sendJson(res, 200, { exports: merged });
        return;
      }

      if (req.method === "POST" && url.pathname === "/api/models/save") {
        try {
          const bodyBuf = await readRawBody(req);
          const contentType = req.headers["content-type"] ?? "";
          let meta: import("../tiny-llm/src/model-store.ts").SavedModelMeta;
          let manifest: ModelManifest;
          let weights: Float32Array;

          if (contentType.includes("application/vnd.tiny-gpt-save.v1")) {
            const decoded = decodeSavePayload(bodyBuf);
            meta = decoded.meta;
            manifest = decoded.manifest;
            weights = decoded.weights;
          } else {
            // Legacy: weights body + JSON in headers (ASCII vocab only)
            meta = JSON.parse(
              req.headers["x-save-meta"] as string,
            ) as import("../tiny-llm/src/model-store.ts").SavedModelMeta;
            manifest = JSON.parse(
              req.headers["x-save-manifest"] as string,
            ) as ModelManifest;
            weights = new Float32Array(
              bodyBuf.buffer,
              bodyBuf.byteOffset,
              bodyBuf.byteLength / 4,
            );
          }

          const folderName =
            (req.headers["x-save-name"] as string | undefined)?.trim() ||
            defaultSaveFolderName(meta.step, meta.label ?? "vi");
          const saved = saveModelToFolder(
            { manifest, weights, meta, label: meta.label },
            folderName,
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
