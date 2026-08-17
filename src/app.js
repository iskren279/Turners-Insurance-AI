import express from "express";
import carValueRouter from "./routes/carValue.js";

const app = express();

app.use(express.json());

app.use(carValueRouter);

export default app;
