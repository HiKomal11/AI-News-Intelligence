// ============================================================
// AI NEWS INTELLIGENCE
// FRONTEND SCRIPT
// ============================================================
//
// LOCAL:
//   - TF-IDF + Logistic Regression
//   - LSTM TFLite / LiteRT
//
// RENDER:
//   - TF-IDF + Logistic Regression ONLY
//   - LSTM automatically hidden
//
// IMPORTANT:
// The frontend controls which model can be selected.
// The backend MUST ALSO reject LSTM requests on Render.
//
// ============================================================


// ============================================================
// GLOBAL STATE
// ============================================================

let activeTab = "text";
let selectedFile = null;
let isAnalyzing = false;


// ============================================================
// DOM HELPER
// ============================================================

function getElement(id) {

    return document.getElementById(id);
}


// ============================================================
// ENVIRONMENT DETECTION
// ============================================================

function getServerEnvironment() {

    const body =
        document.body;

    if (!body) {
        return "";
    }

    return String(
        body.dataset.environment || ""
    )
        .trim()
        .toLowerCase();
}


function isRenderEnvironment() {

    const serverEnvironment =
        getServerEnvironment();


    // --------------------------------------------------------
    // Explicit server environment
    // --------------------------------------------------------

    if (
        serverEnvironment === "render" ||
        serverEnvironment === "production"
    ) {
        return true;
    }


    if (
        serverEnvironment === "local" ||
        serverEnvironment === "development"
    ) {
        return false;
    }


    // --------------------------------------------------------
    // Browser hostname
    // --------------------------------------------------------

    const hostname =
        String(
            window.location.hostname || ""
        )
            .trim()
            .toLowerCase();


    // Render default domains

    if (
        hostname.endsWith(".onrender.com") ||
        hostname === "onrender.com"
    ) {
        return true;
    }


    // --------------------------------------------------------
    // Local development
    // --------------------------------------------------------

    if (
        hostname === "localhost" ||
        hostname === "127.0.0.1" ||
        hostname === "0.0.0.0"
    ) {
        return false;
    }


    // --------------------------------------------------------
    // Private/local network IP
    //
    // Example:
    // 192.168.x.x
    // 10.x.x.x
    // 172.16.x.x - 172.31.x.x
    // --------------------------------------------------------

    if (
        hostname.startsWith("192.168.") ||
        hostname.startsWith("10.")
    ) {
        return false;
    }


    const private172 =
        hostname.match(
            /^172\.(\d+)\./
        );

    if (
        private172 &&
        Number(private172[1]) >= 16 &&
        Number(private172[1]) <= 31
    ) {
        return false;
    }


    // --------------------------------------------------------
    // Unknown domain
    //
    // If the backend explicitly says render/production,
    // it was already handled above.
    //
    // Otherwise assume local/development so that LSTM
    // remains available during local testing.
    // --------------------------------------------------------

    return false;
}


// ============================================================
// ENVIRONMENT NAME
// ============================================================

function getEnvironmentName() {

    return isRenderEnvironment()
        ? "render"
        : "local";
}


// ============================================================
// MODEL NORMALIZATION
// ============================================================

function normalizeModel(model) {

    const normalized =
        String(
            model || "tfidf"
        )
            .trim()
            .toLowerCase();


    if (
        normalized === "lstm"
    ) {
        return "lstm";
    }


    return "tfidf";
}


// ============================================================
// MODEL AVAILABILITY
// ============================================================

function isModelAvailable(model) {

    const normalizedModel =
        normalizeModel(model);


    // --------------------------------------------------------
    // TF-IDF
    // --------------------------------------------------------

    if (
        normalizedModel === "tfidf"
    ) {
        return true;
    }


    // --------------------------------------------------------
    // LSTM
    // --------------------------------------------------------

    if (
        normalizedModel === "lstm"
    ) {
        return !isRenderEnvironment();
    }


    return false;
}


// ============================================================
// GET SELECTED MODEL
// ============================================================

function getSelectedModel() {

    // --------------------------------------------------------
    // Render ALWAYS uses TF-IDF
    // --------------------------------------------------------

    if (
        isRenderEnvironment()
    ) {
        return "tfidf";
    }


    const selected =
        document.querySelector(
            'input[name="model"]:checked'
        );


    if (!selected) {
        return "tfidf";
    }


    const model =
        normalizeModel(
            selected.value
        );


    if (
        !isModelAvailable(model)
    ) {
        return "tfidf";
    }


    return model;
}


// ============================================================
// APPLY MODEL SETTINGS
// ============================================================

function applyEnvironmentModelSettings() {

    const isRender =
        isRenderEnvironment();


    console.log(
        "Environment:",
        getEnvironmentName()
    );


    console.log(
        "Render:",
        isRender
    );


    const modelChoices =
        document.querySelectorAll(
            ".model-choice"
        );


    modelChoices.forEach(choice => {

        const radio =
            choice.querySelector(
                'input[name="model"]'
            );


        if (!radio) {
            return;
        }


        const model =
            normalizeModel(
                radio.value
            );


        // ----------------------------------------------------
        // Render + LSTM
        // ----------------------------------------------------

        if (
            isRender &&
            model === "lstm"
        ) {

            choice.classList.add(
                "hidden"
            );


            radio.checked =
                false;


            radio.disabled =
                true;


            return;
        }


        // ----------------------------------------------------
        // Local + LSTM
        // ----------------------------------------------------

        choice.classList.remove(
            "hidden"
        );


        radio.disabled =
            false;

    });


    // --------------------------------------------------------
    // Render MUST select TF-IDF
    // --------------------------------------------------------

    if (isRender) {

        const tfidfRadio =
            document.querySelector(
                'input[name="model"][value="tfidf"]'
            );


        if (tfidfRadio) {

            tfidfRadio.checked =
                true;


            tfidfRadio.disabled =
                false;
        }
    }


    // --------------------------------------------------------
    // Local:
    // make sure TF-IDF is selected if nothing is selected
    // --------------------------------------------------------

    if (!isRender) {

        const selected =
            document.querySelector(
                'input[name="model"]:checked'
            );


        if (!selected) {

            const tfidfRadio =
                document.querySelector(
                    'input[name="model"][value="tfidf"]'
                );


            if (tfidfRadio) {

                tfidfRadio.checked =
                    true;
            }
        }
    }


    updateModelInformation();
}


// ============================================================
// MODEL INFORMATION
// ============================================================

function updateModelInformation() {

    const modelNote =
        document.querySelector(
            ".model-note"
        );


    if (!modelNote) {
        return;
    }


    if (
        isRenderEnvironment()
    ) {

        modelNote.innerHTML = `
            <strong>Cloud deployment:</strong>
            TF-IDF + Logistic Regression is available.
            The LSTM model is disabled on the Render deployment
            to reduce memory usage.
        `;

        return;
    }


    modelNote.innerHTML = `
        <strong>Local development:</strong>
        Both TF-IDF + Logistic Regression and
        LSTM TFLite/LiteRT are available locally.
        LSTM is loaded only when selected.
    `;
}


// ============================================================
// TAB SWITCHING
// ============================================================

function switchTab(tab) {

    if (
        tab !== "text" &&
        tab !== "file"
    ) {
        tab = "text";
    }


    activeTab =
        tab;


    document
        .querySelectorAll(
            ".switch-button"
        )
        .forEach(button => {

            button.classList.toggle(
                "active",
                button.dataset.tab === tab
            );

        });


    const textPanel =
        getElement(
            "textPanel"
        );


    const filePanel =
        getElement(
            "filePanel"
        );


    if (textPanel) {

        textPanel.classList.toggle(
            "active",
            tab === "text"
        );
    }


    if (filePanel) {

        filePanel.classList.toggle(
            "active",
            tab === "file"
        );
    }


    clearError();
}


// ============================================================
// FILE INPUT INITIALIZATION
// ============================================================

function initializeFileHandling() {

    const fileInput =
        getElement(
            "fileInput"
        );


    const dropZone =
        getElement(
            "dropZone"
        );


    if (fileInput) {

        fileInput.addEventListener(
            "change",
            event => {

                const files =
                    event.target.files;


                if (
                    files &&
                    files.length > 0
                ) {

                    setSelectedFile(
                        files[0]
                    );
                }
            }
        );
    }


    if (dropZone) {

        dropZone.addEventListener(
            "dragover",
            event => {

                event.preventDefault();

                dropZone.classList.add(
                    "dragging"
                );
            }
        );


        dropZone.addEventListener(
            "dragleave",
            event => {

                event.preventDefault();

                dropZone.classList.remove(
                    "dragging"
                );
            }
        );


        dropZone.addEventListener(
            "drop",
            event => {

                event.preventDefault();


                dropZone.classList.remove(
                    "dragging"
                );


                const files =
                    event.dataTransfer?.files;


                if (
                    files &&
                    files.length > 0
                ) {

                    setSelectedFile(
                        files[0]
                    );
                }
            }
        );
    }
}


// ============================================================
// FILE SELECTION
// ============================================================

function setSelectedFile(file) {

    clearError();


    if (!file) {
        return;
    }


    const allowedExtensions = [
        "png",
        "jpg",
        "jpeg",
        "webp",
        "pdf"
    ];


    const filename =
        String(
            file.name || ""
        );


    const extension =
        filename
            .split(".")
            .pop()
            .toLowerCase();


    if (
        !allowedExtensions.includes(
            extension
        )
    ) {

        showError(
            "Unsupported file type. Please upload PNG, JPG, JPEG, WEBP, or PDF."
        );

        return;
    }


    const maxFileSize =
        16 * 1024 * 1024;


    if (
        Number(file.size) >
        maxFileSize
    ) {

        showError(
            "File is larger than 16 MB. Please upload a smaller file."
        );

        return;
    }


    selectedFile =
        file;


    const selectedFileName =
        getElement(
            "selectedFileName"
        );


    if (selectedFileName) {

        selectedFileName.textContent =
            `${filename} (${formatFileSize(file.size)})`;
    }


    const fileIcon =
        document.querySelector(
            ".file-icon"
        );


    if (fileIcon) {

        fileIcon.textContent =
            extension.toUpperCase();
    }


    const selectedFileElement =
        getElement(
            "selectedFile"
        );


    if (selectedFileElement) {

        selectedFileElement.classList.remove(
            "hidden"
        );
    }


    const extractedPreview =
        getElement(
            "extractedPreview"
        );


    const extractedText =
        getElement(
            "extractedText"
        );


    if (extractedPreview) {

        extractedPreview.classList.add(
            "hidden"
        );
    }


    if (extractedText) {

        extractedText.value =
            "";
    }
}


// ============================================================
// REMOVE FILE
// ============================================================

function removeFile() {

    selectedFile =
        null;


    const fileInput =
        getElement(
            "fileInput"
        );


    if (fileInput) {

        fileInput.value =
            "";
    }


    const selectedFileElement =
        getElement(
            "selectedFile"
        );


    if (selectedFileElement) {

        selectedFileElement.classList.add(
            "hidden"
        );
    }


    const extractedPreview =
        getElement(
            "extractedPreview"
        );


    if (extractedPreview) {

        extractedPreview.classList.add(
            "hidden"
        );
    }


    const extractedText =
        getElement(
            "extractedText"
        );


    if (extractedText) {

        extractedText.value =
            "";
    }
}


// ============================================================
// FILE SIZE
// ============================================================

function formatFileSize(bytes) {

    const size =
        Number(bytes);


    if (
        !Number.isFinite(size)
    ) {

        return "Unknown size";
    }


    if (
        size < 1024
    ) {

        return `${size} B`;
    }


    if (
        size <
        1024 * 1024
    ) {

        return `${(
            size / 1024
        ).toFixed(1)} KB`;
    }


    return `${(
        size /
        (1024 * 1024)
    ).toFixed(1)} MB`;
}


// ============================================================
// MAIN ANALYZE
// ============================================================

async function analyze() {

    if (isAnalyzing) {
        return;
    }


    clearError();


    let model =
        getSelectedModel();


    // --------------------------------------------------------
    // HARD Render protection
    // --------------------------------------------------------

    if (
        isRenderEnvironment() &&
        model === "lstm"
    ) {

        console.warn(
            "LSTM blocked on Render. Falling back to TF-IDF."
        );


        model =
            "tfidf";
    }


    if (
        !isModelAvailable(model)
    ) {

        showError(
            "The selected model is not available in this environment."
        );

        return;
    }


    console.log(
        "Environment:",
        getEnvironmentName()
    );


    console.log(
        "Selected model:",
        model
    );


    if (
        activeTab === "text"
    ) {

        await analyzeText(
            model
        );

    } else {

        await analyzeFile(
            model
        );
    }
}


// ============================================================
// TEXT ANALYSIS
// ============================================================

async function analyzeText(model) {

    const textElement =
        getElement(
            "newsText"
        );


    const text =
        textElement
            ? textElement.value.trim()
            : "";


    if (!text) {

        showError(
            "Please enter a news article."
        );

        return;
    }


    const wordCount =
        text
            .split(/\s+/)
            .filter(Boolean)
            .length;


    if (
        wordCount < 5
    ) {

        showError(
            "Please enter a longer news article."
        );

        return;
    }


    if (
        isRenderEnvironment() &&
        model !== "tfidf"
    ) {

        model =
            "tfidf";
    }


    showLoading(true);


    try {

        console.log(
            "POST /predict",
            {
                model
            }
        );


        const response =
            await fetchWithTimeout(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text,
                        model: model
                    })
                },

                120000
            );


        const data =
            await parseServerResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                getServerErrorMessage(
                    data,
                    response.status
                )
            );
        }


        if (
            data &&
            data.error
        ) {

            throw new Error(
                String(data.error)
            );
        }


        validatePredictionResult(
            data
        );


        displayResult(
            data,
            true
        );

    } catch (error) {

        console.error(
            "Text prediction error:",
            error
        );


        showError(
            formatClientError(
                error,
                "Unable to analyze the news article."
            )
        );

    } finally {

        showLoading(false);
    }
}


// ============================================================
// FILE ANALYSIS
// ============================================================

async function analyzeFile(model) {

    if (!selectedFile) {

        showError(
            "Please upload an image or PDF first."
        );

        return;
    }


    if (
        isRenderEnvironment() &&
        model !== "tfidf"
    ) {

        model =
            "tfidf";
    }


    showLoading(true);


    try {

        console.log(
            "POST /upload",
            {
                model
            }
        );


        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        formData.append(
            "model",
            model
        );


        const response =
            await fetchWithTimeout(
                "/upload",
                {
                    method: "POST",

                    headers: {
                        "Accept":
                            "application/json"
                    },

                    body: formData
                },

                120000
            );


        const data =
            await parseServerResponse(
                response
            );


        if (!response.ok) {

            throw new Error(
                getServerErrorMessage(
                    data,
                    response.status
                )
            );
        }


        if (
            data &&
            data.error
        ) {

            throw new Error(
                String(data.error)
            );
        }


        validatePredictionResult(
            data
        );


        displayResult(
            data,
            true
        );


        const extractedText =
            getElement(
                "extractedText"
            );


        const extractedPreview =
            getElement(
                "extractedPreview"
            );


        if (extractedText) {

            extractedText.value =
                data.extracted_text ||
                "";
        }


        if (extractedPreview) {

            extractedPreview.classList.remove(
                "hidden"
            );
        }

    } catch (error) {

        console.error(
            "File prediction error:",
            error
        );


        showError(
            formatClientError(
                error,
                "Unable to analyze the uploaded file."
            )
        );

    } finally {

        showLoading(false);
    }
}


// ============================================================
// FETCH WITH TIMEOUT
// ============================================================

async function fetchWithTimeout(
    url,
    options = {},
    timeout = 120000
) {

    const controller =
        new AbortController();


    const timeoutId =
        setTimeout(
            () => {

                controller.abort();

            },
            timeout
        );


    try {

        return await fetch(
            url,
            {
                ...options,
                signal:
                    controller.signal
            }
        );

    } finally {

        clearTimeout(
            timeoutId
        );
    }
}


// ============================================================
// PARSE SERVER RESPONSE
// ============================================================

async function parseServerResponse(
    response
) {

    const responseText =
        await response.text();


    if (!responseText) {
        return {};
    }


    try {

        return JSON.parse(
            responseText
        );

    } catch (error) {

        console.error(
            "Server returned non-JSON response:",
            responseText
        );


        return {
            error:
                extractHtmlError(
                    responseText
                )
        };
    }
}


// ============================================================
// EXTRACT HTML ERROR
// ============================================================

function extractHtmlError(text) {

    if (!text) {

        return (
            "Server returned an empty response."
        );
    }


    const cleaned =
        String(text)
            .replace(
                /<[^>]*>/g,
                " "
            )
            .replace(
                /\s+/g,
                " "
            )
            .trim();


    if (!cleaned) {

        return (
            "Server returned an invalid response."
        );
    }


    if (
        cleaned.length > 500
    ) {

        return (
            cleaned.substring(
                0,
                500
            ) + "..."
        );
    }


    return cleaned;
}


// ============================================================
// SERVER ERROR MESSAGE
// ============================================================

function getServerErrorMessage(
    data,
    status
) {

    if (
        data &&
        typeof data.error === "string" &&
        data.error.trim()
    ) {

        return data.error.trim();
    }


    switch (
        Number(status)
    ) {

        case 400:

            return (
                "Invalid request. Please check your input."
            );


        case 413:

            return (
                "The uploaded file is too large. Maximum size is 16 MB."
            );


        case 429:

            return (
                "Too many requests. Please wait a moment and try again."
            );


        case 500:

            return (
                "Server error while processing the prediction. Check the server logs."
            );


        case 502:

            return (
                "The prediction server returned 502 Bad Gateway. The backend may have timed out or crashed."
            );


        case 503:

            return (
                "The prediction service is temporarily unavailable."
            );


        case 504:

            return (
                "The prediction server timed out. Please try again."
            );


        default:

            return (
                `Prediction failed. Server returned ${status}.`
            );
    }
}


// ============================================================
// CLIENT ERROR FORMATTER
// ============================================================

function formatClientError(
    error,
    fallback
) {

    if (!error) {
        return fallback;
    }


    if (
        error.name === "AbortError"
    ) {

        return (
            "The server took too long to respond. " +
            "Please try again."
        );
    }


    if (
        error instanceof TypeError
    ) {

        return (
            "Unable to connect to the prediction server. " +
            "Make sure the Flask/Render backend is running."
        );
    }


    if (
        error.message &&
        String(error.message).trim()
    ) {

        return String(
            error.message
        );
    }


    return fallback;
}


// ============================================================
// VALIDATE PREDICTION
// ============================================================

function validatePredictionResult(data) {

    if (!data) {

        throw new Error(
            "The server returned no prediction data."
        );
    }


    if (!data.prediction) {

        throw new Error(
            data.error ||
            "The server did not return a prediction."
        );
    }


    const prediction =
        String(
            data.prediction
        )
            .trim()
            .toUpperCase();


    if (
        prediction !== "REAL" &&
        prediction !== "FAKE"
    ) {

        throw new Error(
            "The server returned an invalid prediction."
        );
    }


    const confidence =
        Number(
            data.confidence
        );


    if (
        !Number.isFinite(
            confidence
        )
    ) {

        throw new Error(
            "The server returned an invalid confidence value."
        );
    }
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(
    data,
    showDownloads = true
) {

    const resultCard =
        getElement(
            "resultCard"
        );


    if (!resultCard) {
        return;
    }


    const prediction =
        String(
            data.prediction || ""
        )
            .trim()
            .toUpperCase();


    const isReal =
        prediction === "REAL";


    const resultIcon =
        getElement(
            "resultIcon"
        );


    if (resultIcon) {

        resultIcon.textContent =
            isReal
                ? "✅"
                : "⚠️";
    }


    const resultTitle =
        getElement(
            "resultTitle"
        );


    if (resultTitle) {

        resultTitle.textContent =
            data.result ||
            (
                isReal
                    ? "Real News"
                    : "Fake News"
            );
    }


    let confidence =
        Number(
            data.confidence
        );


    if (
        !Number.isFinite(
            confidence
        )
    ) {

        confidence =
            0;
    }


    confidence =
        Math.max(
            0,
            Math.min(
                100,
                confidence
            )
        );


    const confidenceElement =
        getElement(
            "confidence"
        );


    if (confidenceElement) {

        confidenceElement.textContent =
            `${confidence.toFixed(2)}%`;
    }


    const confidenceBar =
        getElement(
            "confidenceBar"
        );


    if (confidenceBar) {

        confidenceBar.style.width =
            `${confidence}%`;
    }


    const resultModel =
        getElement(
            "resultModel"
        );


    if (resultModel) {

        resultModel.textContent =
            data.model ||
            (
                isRenderEnvironment()
                    ? "TF-IDF"
                    : "Unknown"
            );
    }


    const resultWords =
        getElement(
            "resultWords"
        );


    if (resultWords) {

        resultWords.textContent =
            data.word_count ?? 0;
    }


    const resultCharacters =
        getElement(
            "resultCharacters"
        );


    if (resultCharacters) {

        resultCharacters.textContent =
            data.character_count ?? 0;
    }


    let sourceText =
        "Source: Typed Text";


    if (data.source_type) {

        sourceText =
            `Source: ${data.source_type}`;
    }


    if (data.filename) {

        sourceText +=
            ` • ${data.filename}`;
    }


    const sourceInfo =
        getElement(
            "sourceInfo"
        );


    if (sourceInfo) {

        sourceInfo.textContent =
            sourceText;
    }


    resultCard.classList.remove(
        "real-result",
        "fake-result"
    );


    resultCard.classList.add(
        isReal
            ? "real-result"
            : "fake-result"
    );


    const downloadSection =
        getElement(
            "downloadSection"
        );


    const downloadPdf =
        getElement(
            "downloadPdf"
        );


    const downloadPng =
        getElement(
            "downloadPng"
        );


    if (
        showDownloads &&
        data.pdf_url &&
        data.png_url
    ) {

        if (downloadPdf) {

            downloadPdf.href =
                data.pdf_url;
        }


        if (downloadPng) {

            downloadPng.href =
                data.png_url;
        }


        if (downloadSection) {

            downloadSection.classList.remove(
                "hidden"
            );
        }

    } else {

        if (downloadSection) {

            downloadSection.classList.add(
                "hidden"
            );
        }
    }


    resultCard.classList.remove(
        "hidden"
    );


    setTimeout(
        () => {

            if (
                typeof resultCard.scrollIntoView ===
                "function"
            ) {

                resultCard.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

        },
        50
    );
}


// ============================================================
// LOADING
// ============================================================

function showLoading(show) {

    isAnalyzing =
        Boolean(show);


    const loading =
        getElement(
            "loading"
        );


    if (loading) {

        loading.classList.toggle(
            "hidden",
            !show
        );
    }


    const button =
        getElement(
            "analyzeButton"
        );


    if (button) {

        button.disabled =
            show;


        if (show) {

            if (
                !button.dataset.originalText
            ) {

                button.dataset.originalText =
                    button.innerHTML;
            }


            button.innerHTML = `
                Analyzing...
                <span>⏳</span>
            `;

        } else {

            button.innerHTML =
                button.dataset.originalText ||
                `
                    Analyze News
                    <span>→</span>
                `;
        }
    }


    document
        .querySelectorAll(
            'input[name="model"]'
        )
        .forEach(radio => {

            const model =
                normalizeModel(
                    radio.value
                );


            // Render LSTM stays disabled

            if (
                isRenderEnvironment() &&
                model === "lstm"
            ) {

                radio.disabled =
                    true;

                return;
            }


            radio.disabled =
                show;
        });
}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    const error =
        getElement(
            "error"
        );


    if (!error) {

        console.error(
            message
        );

        return;
    }


    error.textContent =
        String(
            message ||
            "An unexpected error occurred."
        );


    error.classList.remove(
        "hidden"
    );


    setTimeout(
        () => {

            if (
                typeof error.scrollIntoView ===
                "function"
            ) {

                error.scrollIntoView({
                    behavior: "smooth",
                    block: "nearest"
                });
            }

        },
        50
    );
}


// ============================================================
// CLEAR ERROR
// ============================================================

function clearError() {

    const error =
        getElement(
            "error"
        );


    if (error) {

        error.textContent =
            "";


        error.classList.add(
            "hidden"
        );
    }
}


// ============================================================
// CLEAR EVERYTHING
// ============================================================

function clearAll() {

    const newsText =
        getElement(
            "newsText"
        );


    if (newsText) {

        newsText.value =
            "";
    }


    removeFile();


    const resultCard =
        getElement(
            "resultCard"
        );


    if (resultCard) {

        resultCard.classList.add(
            "hidden"
        );


        resultCard.classList.remove(
            "real-result",
            "fake-result"
        );
    }


    const downloadSection =
        getElement(
            "downloadSection"
        );


    if (downloadSection) {

        downloadSection.classList.add(
            "hidden"
        );
    }


    const confidenceBar =
        getElement(
            "confidenceBar"
        );


    if (confidenceBar) {

        confidenceBar.style.width =
            "0%";
    }


    const confidence =
        getElement(
            "confidence"
        );


    if (confidence) {

        confidence.textContent =
            "0%";
    }


    clearError();


    switchTab(
        "text"
    );


    applyEnvironmentModelSettings();
}


// ============================================================
// THEME
// ============================================================

function setTheme(theme) {

    const body =
        document.body;


    if (!body) {
        return;
    }


    const lightButton =
        getElement(
            "lightMode"
        );


    const darkButton =
        getElement(
            "darkMode"
        );


    if (
        theme === "dark"
    ) {

        body.classList.add(
            "dark-mode"
        );


        if (darkButton) {

            darkButton.classList.add(
                "active"
            );
        }


        if (lightButton) {

            lightButton.classList.remove(
                "active"
            );
        }

    } else {

        body.classList.remove(
            "dark-mode"
        );


        if (lightButton) {

            lightButton.classList.add(
                "active"
            );
        }


        if (darkButton) {

            darkButton.classList.remove(
                "active"
            );
        }


        theme =
            "light";
    }


    try {

        localStorage.setItem(
            "aiNewsTheme",
            theme
        );

    } catch (error) {

        console.warn(
            "Unable to save theme preference.",
            error
        );
    }
}


// ============================================================
// LOAD THEME
// ============================================================

function loadTheme() {

    let savedTheme =
        "light";


    try {

        savedTheme =
            localStorage.getItem(
                "aiNewsTheme"
            ) ||
            "light";

    } catch (error) {

        console.warn(
            "Unable to read saved theme.",
            error
        );
    }


    if (
        savedTheme === "dark"
    ) {

        setTheme(
            "dark"
        );

    } else {

        setTheme(
            "light"
        );
    }
}


// ============================================================
// MODEL CHANGE HANDLER
// ============================================================

function initializeModelHandling() {

    document
        .querySelectorAll(
            'input[name="model"]'
        )
        .forEach(radio => {

            radio.addEventListener(
                "change",
                () => {

                    clearError();


                    if (
                        isRenderEnvironment() &&
                        normalizeModel(
                            radio.value
                        ) === "lstm"
                    ) {

                        const tfidfRadio =
                            document.querySelector(
                                'input[name="model"][value="tfidf"]'
                            );


                        if (tfidfRadio) {

                            tfidfRadio.checked =
                                true;
                        }


                        showError(
                            "LSTM is available only in the local version. TF-IDF has been selected."
                        );


                        return;
                    }


                    console.log(
                        "Environment:",
                        getEnvironmentName()
                    );


                    console.log(
                        "Model changed:",
                        getSelectedModel()
                    );
                }
            );
        });
}


// ============================================================
// KEYBOARD SHORTCUT
// ============================================================
//
// Ctrl + Enter = Analyze
//
// ============================================================

function initializeKeyboardShortcut() {

    document.addEventListener(
        "keydown",
        event => {

            if (
                event.ctrlKey &&
                event.key === "Enter"
            ) {

                event.preventDefault();


                if (!isAnalyzing) {

                    analyze();
                }
            }
        }
    );
}


// ============================================================
// PREVENT ACCIDENTAL PAGE SUBMISSION
// ============================================================

function initializeBeforeUnload() {

    window.addEventListener(
        "beforeunload",
        event => {

            if (isAnalyzing) {

                event.preventDefault();

                event.returnValue =
                    "";
            }
        }
    );
}


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        console.log(
            "AI News Intelligence frontend loaded."
        );


        console.log(
            "Environment:",
            getEnvironmentName()
        );


        console.log(
            "Hostname:",
            window.location.hostname
        );


        console.log(
            "Server environment:",
            getServerEnvironment()
        );


        console.log(
            "LSTM available:",
            isModelAvailable("lstm")
        );


        console.log(
            "TF-IDF available:",
            isModelAvailable("tfidf")
        );


        // ----------------------------------------------------
        // Theme
        // ----------------------------------------------------

        loadTheme();


        // ----------------------------------------------------
        // Model restrictions
        // ----------------------------------------------------

        applyEnvironmentModelSettings();


        // ----------------------------------------------------
        // File handling
        // ----------------------------------------------------

        initializeFileHandling();


        // ----------------------------------------------------
        // Model handling
        // ----------------------------------------------------

        initializeModelHandling();


        // ----------------------------------------------------
        // Keyboard
        // ----------------------------------------------------

        initializeKeyboardShortcut();


        // ----------------------------------------------------
        // Before unload
        // ----------------------------------------------------

        initializeBeforeUnload();


        // ----------------------------------------------------
        // Default tab
        // ----------------------------------------------------

        switchTab(
            "text"
        );

    }
);