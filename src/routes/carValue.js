import express from "express";
import calculateCarValue from "../services/carValue.js";

const router = express.Router();

router.post("/api/car-value", (req, res) => {
  const { model, year } = req.body;

  try {
    const carValue = calculateCarValue(model, year);

    res.status(200).json({
      car_value: carValue,
    });
  } catch (error) {
    res.status(400).json({
      error: "there is an error",
    });
  }
});

export default router;
