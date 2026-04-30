const imageInput = document.getElementById("image-input");
const analyzeButton = document.getElementById("analyze-button");
const previewImage = document.getElementById("preview-image");
const dishTitle = document.getElementById("dish-title");
const dishSubtitle = document.getElementById("dish-subtitle");
const statusText = document.getElementById("status");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chipButtons = Array.from(document.querySelectorAll(".chips button"));

const state = {
    selectedFile: null,
    prediction: null,
    previewUrl: null,
};

function setStatus(text, tone = "info") {
    statusText.textContent = text;
    statusText.dataset.tone = tone;
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;",
    }[char]));
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({ message: "Request failed." }));
    if (!response.ok || data.ok === false) {
        throw new Error(data.message || "Request failed.");
    }
    return data;
}

function appendMessage(role, text) {
    const item = document.createElement("article");
    item.className = `bubble ${role}`;
    item.innerHTML = `<strong>${role === "assistant" ? "Food AI" : "You"}</strong><p>${escapeHtml(text)}</p>`;
    chatLog.appendChild(item);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function resetChat() {
    chatLog.innerHTML = "";
    appendMessage("assistant", "Upload an image and run analysis to start.");
}

imageInput.addEventListener("change", () => {
    state.selectedFile = imageInput.files[0] || null;
    state.prediction = null;
    resetChat();

    if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
    }
    if (state.selectedFile) {
        state.previewUrl = URL.createObjectURL(state.selectedFile);
        previewImage.src = state.previewUrl;
        dishTitle.textContent = "Image selected";
        dishSubtitle.textContent = "Run analysis to identify the dish.";
        setStatus("Image selected.", "success");
    }
});

analyzeButton.addEventListener("click", async () => {
    if (!state.selectedFile) {
        setStatus("Upload an image first.", "error");
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";
    setStatus("Running FITFoodNet and generating feedback...", "loading");

    try {
        const formData = new FormData();
        formData.append("image", state.selectedFile);
        const prediction = await fetchJson("/api/predict", { method: "POST", body: formData });
        state.prediction = prediction;

        const top = prediction.top_prediction;
        dishTitle.textContent = top ? top.label : "Prediction unavailable";
        dishSubtitle.textContent = top ? `Confidence ${(top.confidence * 100).toFixed(2)}%` : "";

        const aiReply = await fetchJson("/api/food-ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: "Describe the dish, likely ingredients, and one nutrition suggestion.",
                dish_name: top ? top.label : null,
                predictions: prediction.predictions ? prediction.predictions.slice(0, 3) : [],
            }),
        });

        chatLog.innerHTML = "";
        appendMessage("assistant", aiReply.answer);
        setStatus("Analysis complete.", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Run Analysis";
    }
});

async function askFoodAi(message) {
    if (!state.prediction || !state.prediction.top_prediction) {
        setStatus("Run image analysis first.", "error");
        return;
    }

    appendMessage("user", message);
    setStatus("Generating response...", "loading");

    try {
        const reply = await fetchJson("/api/food-ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message,
                dish_name: state.prediction.top_prediction.label,
                predictions: state.prediction.predictions ? state.prediction.predictions.slice(0, 3) : [],
            }),
        });
        appendMessage("assistant", reply.answer);
        setStatus("Ready.", "success");
    } catch (error) {
        setStatus(error.message, "error");
    }
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message) {
        return;
    }
    chatInput.value = "";
    await askFoodAi(message);
});

chipButtons.forEach((button) => {
    button.addEventListener("click", () => {
        chatInput.value = button.dataset.prompt || "";
        chatForm.requestSubmit();
    });
});

async function bootstrap() {
    resetChat();
    try {
        const health = await fetchJson("/api/health");
        setStatus(health.details || "Service ready.", health.model_ready ? "success" : "info");
    } catch (error) {
        setStatus(error.message, "error");
    }
}

bootstrap();
