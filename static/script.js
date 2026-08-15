let activeTab = "text";
let selectedFile = null;


// ============================================================
// TAB SWITCHING
// ============================================================

function switchTab(tab) {

    activeTab = tab;

    // Update highlighted tab
    document
        .querySelectorAll(".switch-button")
        .forEach(button => {

            button.classList.toggle(
                "active",
                button.dataset.tab === tab
            );

        });


    // Switch content panel
    document
        .getElementById("textPanel")
        .classList.toggle(
            "active",
            tab === "text"
        );


    document
        .getElementById("filePanel")
        .classList.toggle(
            "active",
            tab === "file"
        );

}

// ============================================================
// FILE INPUT
// ============================================================

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");

if (fileInput) {

    fileInput.addEventListener("change", event => {

        const file = event.target.files[0];

        if (file) {
            setSelectedFile(file);
        }

    });

}


// ============================================================
// DRAG & DROP
// ============================================================

if (dropZone) {

    dropZone.addEventListener("dragover", event => {

        event.preventDefault();

        dropZone.classList.add("dragging");

    });


    dropZone.addEventListener("dragleave", () => {

        dropZone.classList.remove("dragging");

    });


    dropZone.addEventListener("drop", event => {

        event.preventDefault();

        dropZone.classList.remove("dragging");

        const file = event.dataTransfer.files[0];

        if (file) {
            setSelectedFile(file);
        }

    });

}


// ============================================================
// FILE SELECTION
// ============================================================

function setSelectedFile(file) {

    const allowedExtensions = [
        "png",
        "jpg",
        "jpeg",
        "webp",
        "pdf"
    ];

    const extension = file.name
        .split(".")
        .pop()
        .toLowerCase();


    if (!allowedExtensions.includes(extension)) {

        showError(
            "Unsupported file type. Please upload PNG, JPG, JPEG, WEBP, or PDF."
        );

        return;

    }


    if (file.size > 16 * 1024 * 1024) {

        showError(
            "File is larger than 16 MB."
        );

        return;

    }


    selectedFile = file;


    document.getElementById(
        "selectedFileName"
    ).textContent =
        `${file.name} (${formatFileSize(file.size)})`;


    document.getElementById(
        "selectedFile"
    ).classList.remove("hidden");


    clearError();
}


// ============================================================
// REMOVE FILE
// ============================================================

function removeFile() {

    selectedFile = null;

    if (fileInput) {
        fileInput.value = "";
    }

    document.getElementById(
        "selectedFile"
    ).classList.add("hidden");


    document.getElementById(
        "extractedPreview"
    ).classList.add("hidden");


    document.getElementById(
        "extractedText"
    ).value = "";
}


// ============================================================
// FILE SIZE
// ============================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


// ============================================================
// GET SELECTED MODEL
// ============================================================

function getSelectedModel() {

    const selected = document.querySelector(
        'input[name="model"]:checked'
    );

    return selected
        ? selected.value
        : "tfidf";
}


// ============================================================
// MAIN ANALYZE FUNCTION
// ============================================================

async function analyze() {

    clearError();

    const model = getSelectedModel();

    if (activeTab === "text") {

        await analyzeText(model);

    } else {

        await analyzeFile(model);

    }
}


// ============================================================
// TEXT ANALYSIS
// ============================================================

async function analyzeText(model) {

    const text = document
        .getElementById("newsText")
        .value
        .trim();


    if (!text) {

        showError(
            "Please enter a news article."
        );

        return;

    }


    if (text.split(/\s+/).length < 5) {

        showError(
            "Please enter a longer news article."
        );

        return;

    }


    showLoading(true);


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    text: text,
                    model: model
                })
            }
        );


        // Read response as text first
        const responseText =
            await response.text();


        let data = {};


        // Safely parse JSON
        if (responseText) {

            try {

                data = JSON.parse(
                    responseText
                );

            } catch (parseError) {

                console.error(
                    "Invalid JSON response:",
                    responseText
                );

                throw new Error(
                    `Server returned an invalid response (${response.status}).`
                );

            }

        }


        // Handle HTTP errors
        if (!response.ok) {

            throw new Error(
                data.error ||
                `Prediction failed. Server returned ${response.status}.`
            );

        }


        displayResult(
            data,
            true
        );


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        showError(
            error.message ||
            "Unable to analyze the news article."
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


    showLoading(true);


    try {

        const formData = new FormData();

        formData.append(
            "file",
            selectedFile
        );

        formData.append(
            "model",
            model
        );


        const response = await fetch(
            "/upload",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "File analysis failed."
            );

        }


        displayResult(
            data,
            true
        );


        document.getElementById(
            "extractedText"
        ).value =
            data.extracted_text || "";


        document.getElementById(
            "extractedPreview"
        ).classList.remove(
            "hidden"
        );


    } catch (error) {

        showError(
            error.message
        );

    } finally {

        showLoading(false);

    }
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(
    data,
    showDownloads
) {

    const resultCard =
        document.getElementById(
            "resultCard"
        );


    const isReal =
        data.prediction === "REAL";


    document.getElementById(
        "resultIcon"
    ).textContent =
        isReal
            ? "✅"
            : "⚠️";


    document.getElementById(
        "resultTitle"
    ).textContent =
        data.result;


    document.getElementById(
        "confidence"
    ).textContent =
        `${data.confidence}%`;


    document.getElementById(
        "confidenceBar"
    ).style.width =
        `${data.confidence}%`;


    document.getElementById(
        "resultModel"
    ).textContent =
        data.model;


    document.getElementById(
        "resultWords"
    ).textContent =
        data.word_count;


    document.getElementById(
        "resultCharacters"
    ).textContent =
        data.character_count;


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


    document.getElementById(
        "sourceInfo"
    ).textContent =
        sourceText;


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
        document.getElementById(
            "downloadSection"
        );


    if (
        showDownloads &&
        data.pdf_url &&
        data.png_url
    ) {

        document.getElementById(
            "downloadPdf"
        ).href =
            data.pdf_url;


        document.getElementById(
            "downloadPng"
        ).href =
            data.png_url;


        downloadSection.classList.remove(
            "hidden"
        );

    } else {

        downloadSection.classList.add(
            "hidden"
        );

    }


    resultCard.classList.remove(
        "hidden"
    );


    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// ============================================================
// LOADING
// ============================================================

function showLoading(show) {

    document.getElementById(
        "loading"
    ).classList.toggle(
        "hidden",
        !show
    );


    const button =
        document.getElementById(
            "analyzeButton"
        );


    if (button) {
        button.disabled = show;
    }
}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    const error =
        document.getElementById(
            "error"
        );


    error.textContent =
        message;


    error.classList.remove(
        "hidden"
    );
}


function clearError() {

    document.getElementById(
        "error"
    ).classList.add(
        "hidden"
    );
}


// ============================================================
// CLEAR EVERYTHING
// ============================================================

function clearAll() {

    document.getElementById(
        "newsText"
    ).value = "";


    removeFile();


    document.getElementById(
        "resultCard"
    ).classList.add(
        "hidden"
    );


    document.getElementById(
        "extractedPreview"
    ).classList.add(
        "hidden"
    );


    clearError();


    switchTab("text");
}

// ============================================================
// THEME
// ============================================================

function setTheme(theme) {

    const body = document.body;

    const lightButton =
        document.getElementById("lightMode");

    const darkButton =
        document.getElementById("darkMode");


    if (theme === "dark") {

        body.classList.add("dark-mode");

        darkButton.classList.add("active");
        lightButton.classList.remove("active");

    } else {

        body.classList.remove("dark-mode");

        lightButton.classList.add("active");
        darkButton.classList.remove("active");

    }


    localStorage.setItem(
        "aiNewsTheme",
        theme
    );
}


// ============================================================
// LOAD SAVED THEME
// ============================================================

function loadTheme() {

    const savedTheme =
        localStorage.getItem(
            "aiNewsTheme"
        );


    if (savedTheme === "dark") {

        setTheme("dark");

    } else {

        setTheme("light");

    }

}


document.addEventListener(
    "DOMContentLoaded",
    loadTheme
);