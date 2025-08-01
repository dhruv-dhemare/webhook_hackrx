const express = require("express");
const axios = require("axios");
const dotenv = require("dotenv");

dotenv.config();
const app = express();
app.use(express.json());

// ✅ Webhook endpoint
app.post("/api/v1/hackrx/run", async (req, res) => {
  try {
    // Incoming payload (documents + questions)
    const { documents, questions } = req.body;

    if (!documents || !questions) {
      return res.status(400).json({ error: "Missing documents or questions" });
    }

    // Call your internal backend API
    const response = await axios.post(
      process.env.BACKEND_URL,
      { documents, questions },
      {
        headers: {
          "Authorization": `Bearer ${process.env.AUTH_TOKEN}`,
          "Content-Type": "application/json",
        },
      }
    );

    // ✅ Return backend response directly to the webhook caller
    res.json(response.data);

  } catch (error) {
    console.error("❌ Webhook Error:", error.response?.data || error.message);
    res.status(500).json({ error: "Failed to process webhook" });
  }
});

// Start server
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`🚀 Webhook server running at http://localhost:${PORT}/api/v1/hackrx/run`);
});
