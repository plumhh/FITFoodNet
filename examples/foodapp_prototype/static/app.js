const appRoot = document.getElementById("app");
const homeView = document.getElementById("home-view");
const chatTitle = document.getElementById("chat-title");
const imageInput = document.getElementById("image-input");
const cameraButton = document.getElementById("camera-button");
const uploadButton = document.getElementById("upload-button");
const fileLabel = document.getElementById("file-label");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const menuButton = document.getElementById("menu-button");
const newButton = document.getElementById("new-button");
const drawer = document.getElementById("drawer");
const closeDrawerButton = document.getElementById("close-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");

const INITIAL_PROMPT = "Describe this food image and provide one nutrition-oriented suggestion.";

const state = {
    selectedFile: null,
    previewUrl: null,
    prediction: null,
    busy: false,
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;",
    }[char]));
}

function cleanText(value) {
    return String(value || "")
        .replace(/\*\*/g, "")
        .replace(/\r/g, "")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const raw = await response.text();
    let data;

    try {
        data = JSON.parse(raw);
    } catch {
        data = { message: raw || "Request failed." };
    }

    if (!response.ok || data.ok === false) {
        throw new Error(data.message || "Request failed.");
    }

    return data;
}

function revokePreviewUrl() {
    if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
        state.previewUrl = null;
    }
}

function ensureChatMode() {
    homeView.classList.add("is-hidden");
    chatTitle.hidden = false;
    chatLog.classList.remove("is-empty");
}

function resetApp() {
    revokePreviewUrl();
    state.selectedFile = null;
    state.prediction = null;
    state.busy = false;
    imageInput.value = "";
    fileLabel.textContent = "No image selected";
    chatInput.value = "";
    chatLog.innerHTML = "";
    chatLog.classList.add("is-empty");
    homeView.classList.remove("is-hidden");
    chatTitle.hidden = true;
    setBusy(false);
    closeDrawer();
}

function setBusy(isBusy) {
    state.busy = isBusy;
    sendButton.disabled = isBusy;
    imageInput.disabled = isBusy;
    cameraButton.disabled = isBusy;
    uploadButton.disabled = isBusy;
    sendButton.textContent = isBusy ? "..." : "Send";
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    });
}

function appendUserText(text) {
    chatLog.classList.remove("is-empty");
    const item = document.createElement("article");
    item.className = "message user";
    item.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    chatLog.appendChild(item);
    scrollToBottom();
}

function appendUserImage(src) {
    chatLog.classList.remove("is-empty");
    const image = document.createElement("img");
    image.className = "image-message";
    image.src = src;
    image.alt = "Uploaded food image";
    chatLog.appendChild(image);
    scrollToBottom();
}

function appendAssistantText(text) {
    chatLog.classList.remove("is-empty");
    const item = document.createElement("article");
    item.className = "message assistant";
    item.innerHTML = `<div class="assistant-bubble">${escapeHtml(cleanText(text)).replace(/\n/g, "<br>")}</div>`;
    chatLog.appendChild(item);
    scrollToBottom();
}

function appendThinking() {
    chatLog.classList.remove("is-empty");
    const item = document.createElement("article");
    item.className = "message assistant";
    item.dataset.thinking = "true";
    item.innerHTML = `<div class="thinking-bubble">Thinking...</div>`;
    chatLog.appendChild(item);
    scrollToBottom();
    return item;
}

function removeThinking(node) {
    if (node && node.parentNode) {
        node.parentNode.removeChild(node);
    }
}

function parseInitialAnswer(answer) {
    const cleaned = cleanText(answer);
    const visualMatch = cleaned.match(/(?:visual description|food description)\s*:?\s*([\s\S]*?)(?=(?:nutrition(?:-oriented)?\s*(?:suggestion|note)|$))/i);
    const nutritionMatch = cleaned.match(/(?:nutrition(?:-oriented)?\s*(?:suggestion|note))\s*:?\s*([\s\S]*)/i);

    let visual = visualMatch ? visualMatch[1].trim() : "";
    let nutrition = nutritionMatch ? nutritionMatch[1].trim() : "";

    if (!visual || !nutrition) {
        const parts = cleaned.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
        visual = visual || parts[0] || cleaned;
        nutrition = nutrition || parts.slice(1).join(" ") || "Use this information as a general nutrition-oriented reference, not as medical advice.";
    }

    return {
        visual: visual.replace(/^[:\-\s]+/, ""),
        nutrition: nutrition.replace(/^[:\-\s]+/, ""),
    };
}

function appendAnalysisCard(prediction, aiAnswer) {
    chatLog.classList.remove("is-empty");
    const top = prediction?.top_prediction || null;
    const sections = parseInitialAnswer(aiAnswer);
    const card = document.createElement("article");
    card.className = "message assistant";
    card.innerHTML = `
        <div class="analysis-card">
            <section class="analysis-section">
                <h3>Food recognition</h3>
                <div class="metric-row">
                    <span>Predicted category</span>
                    <strong>${escapeHtml(top?.label || "Unknown")}</strong>
                </div>
                <div class="metric-row">
                    <span>Confidence</span>
                    <strong>${top ? `${(Number(top.confidence || 0) * 100).toFixed(2)}%` : "--"}</strong>
                </div>
            </section>
            <section class="analysis-section">
                <h4>Visual description</h4>
                <p>${escapeHtml(sections.visual)}</p>
            </section>
            <section class="analysis-section">
                <h4>Nutrition note</h4>
                <p>${escapeHtml(sections.nutrition)}</p>
            </section>
        </div>
    `;
    chatLog.appendChild(card);
    scrollToBottom();
}

function chooseImage(useCamera) {
    if (state.busy) {
        return;
    }
    if (useCamera) {
        imageInput.setAttribute("capture", "environment");
        cameraButton.classList.add("active");
        uploadButton.classList.remove("active");
    } else {
        imageInput.removeAttribute("capture");
        uploadButton.classList.add("active");
        cameraButton.classList.remove("active");
    }
    imageInput.click();
}

async function analyzeSelectedImage(message) {
    if (!state.selectedFile) {
        appendAssistantText("Please choose or capture a food image first.");
        return;
    }

    setBusy(true);
    ensureChatMode();

    if (!state.previewUrl) {
        state.previewUrl = URL.createObjectURL(state.selectedFile);
    }

    appendUserImage(state.previewUrl);
    appendUserText(message);
    const thinking = appendThinking();

    try {
        const formData = new FormData();
        formData.append("image", state.selectedFile);
        const prediction = await fetchJson("/api/predict", {
            method: "POST",
            body: formData,
        });

        const aiReply = await fetchJson("/api/food-ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: [
                    "Using only the predicted labels and confidence scores, provide two concise sections.",
                    "Section 1 heading: Visual description.",
                    "Section 2 heading: Nutrition note.",
                    "Do not infer hidden allergens, clinical suitability, or exact nutrient values.",
                    `User request: ${message}`,
                ].join(" "),
                dish_name: prediction.top_prediction ? prediction.top_prediction.label : null,
                predictions: prediction.predictions ? prediction.predictions.slice(0, 3) : [],
            }),
        });

        state.prediction = prediction;
        removeThinking(thinking);
        appendAnalysisCard(prediction, aiReply.answer);
    } catch (error) {
        removeThinking(thinking);
        appendAssistantText(error.message);
    } finally {
        setBusy(false);
    }
}

async function askFollowUp(message) {
    if (!state.prediction?.top_prediction) {
        appendAssistantText("Please analyze a food image first, then ask a follow-up question.");
        return;
    }

    setBusy(true);
    appendUserText(message);
    const thinking = appendThinking();

    try {
        const aiReply = await fetchJson("/api/food-ai", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: [
                    "Answer the user directly in a concise conversational style.",
                    "Do not use headings unless the user asks for a list or report.",
                    "Do not infer hidden allergens, clinical suitability, or exact nutrient values.",
                    `User question: ${message}`,
                ].join(" "),
                dish_name: state.prediction.top_prediction.label,
                predictions: state.prediction.predictions ? state.prediction.predictions.slice(0, 3) : [],
            }),
        });

        removeThinking(thinking);
        appendAssistantText(aiReply.answer);
    } catch (error) {
        removeThinking(thinking);
        appendAssistantText(error.message);
    } finally {
        setBusy(false);
    }
}

function openDrawer() {
    drawer.classList.add("is-open");
    drawer.setAttribute("aria-hidden", "false");
    drawerBackdrop.hidden = false;
}

function closeDrawer() {
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    drawerBackdrop.hidden = true;
}

cameraButton.addEventListener("click", () => chooseImage(true));
uploadButton.addEventListener("click", () => chooseImage(false));
menuButton.addEventListener("click", openDrawer);
closeDrawerButton.addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", closeDrawer);
newButton.addEventListener("click", resetApp);

drawer.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) {
        return;
    }
    if (button.dataset.action === "new") {
        resetApp();
        return;
    }
    closeDrawer();
});

imageInput.addEventListener("change", () => {
    revokePreviewUrl();
    state.selectedFile = imageInput.files[0] || null;
    fileLabel.textContent = state.selectedFile ? state.selectedFile.name : "No image selected";
});

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy) {
        return;
    }

    const message = chatInput.value.trim() || INITIAL_PROMPT;
    chatInput.value = "";

    if (!state.prediction) {
        await analyzeSelectedImage(message);
    } else {
        await askFollowUp(message);
    }
});

async function bootstrap() {
    resetApp();
    try {
        await fetchJson("/api/health");
    } catch (error) {
        appendAssistantText(error.message);
    }
}

bootstrap();
