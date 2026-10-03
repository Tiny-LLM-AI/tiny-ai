import * as tf from "@tensorflow/tfjs";

/** Browser chat and training require WebGL; never silently fall back to CPU. */
export async function initTfBackend(): Promise<string> {
  try {
    await import("@tensorflow/tfjs-backend-webgl");
    tf.env().set("WEBGL_CPU_FORWARD", false);
    if (!(await tf.setBackend("webgl"))) throw new Error("WebGL unavailable");
    await tf.ready();
    if (tf.getBackend() !== "webgl") throw new Error("WebGL not selected");
    return tf.getBackend();
  } catch (error) {
    throw new Error(`Không khởi tạo được GPU WebGL. Bật hardware acceleration trong trình duyệt và kiểm tra driver GPU. ${error instanceof Error ? error.message : String(error)}`);
  }
}
