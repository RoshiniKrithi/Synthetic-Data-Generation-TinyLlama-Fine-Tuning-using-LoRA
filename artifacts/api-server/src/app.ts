import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { createProxyMiddleware } from "http-proxy-middleware";
import router from "./routes";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

// Forward everything else to the Gradio app on port 8000
app.use(
  "/",
  createProxyMiddleware({
    target: "http://localhost:8000",
    changeOrigin: true,
    ws: true,
    on: {
      error: (_err, _req, res) => {
        if (res && "writeHead" in res) {
          (res as import("http").ServerResponse).writeHead(502);
          (res as import("http").ServerResponse).end(
            "Gradio app is starting up — please wait a moment and refresh.",
          );
        }
      },
    },
  }),
);

export default app;
